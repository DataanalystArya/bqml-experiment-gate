import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, request

app = Flask(__name__)

SAFE_INT_MAX = 9007199254740991
RUNS = {}

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def compact(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    )


def utf8(s):
    return s.encode("utf-8")


def sort_codes(codes):
    return sorted(set(codes), key=utf8)


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_INT_MAX
    )


def finite(x):
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(float(x))


def parse_time(value):
    if not isinstance(value, str):
        return None

    m = TIME_RE.fullmatch(value)
    if not m:
        return None

    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    second = int(m.group(6))

    frac = m.group(7)
    offset = m.group(8)

    microsecond = (
        0 if frac is None
        else int(frac.ljust(3, "0")) * 1000
    )

    try:
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond
        )
    except ValueError:
        return None

    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1
        oh = int(offset[1:3])
        om = int(offset[4:6])

        if oh > 14:
            return None

        if om > 59:
            return None

        if oh == 14 and om != 0:
            return None

        tz = timezone(
            sign * timedelta(
                hours=oh,
                minutes=om
            )
        )

    return dt.replace(tzinfo=tz).astimezone(timezone.utc)


def utc_time(value):
    dt = parse_time(value)

    if dt is None:
        return None

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


def invalid():
    return jsonify({"error": "INVALID_INPUT"}), 400


# ============================================================
# SELECT
# ============================================================

def valid_feature(feature):
    if not isinstance(feature, dict):
        return False

    if set(feature.keys()) != {"value", "availableAt"}:
        return False

    if not isinstance(feature["value"], str):
        return False

    if parse_time(feature["availableAt"]) is None:
        return False

    return True


def valid_select_row(row):
    if not isinstance(row, dict):
        return False

    expected = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    }

    if set(row.keys()) != expected:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    event_time = parse_time(row["eventTime"])
    prediction_time = parse_time(row["predictionTime"])

    if event_time is None or prediction_time is None:
        return False

    if not safe_int(row["version"]):
        return False

    if row["split"] not in {"TRAIN", "EVAL"}:
        return False

    if not isinstance(row["features"], dict):
        return False

    for name, feature in row["features"].items():
        if not isinstance(name, str):
            return False

        if not valid_feature(feature):
            return False

        available = parse_time(feature["availableAt"])

        if available > prediction_time:
            return False

    return True


def valid_trial(trial):
    if not isinstance(trial, dict):
        return False

    if set(trial.keys()) != {
        "trialId",
        "status",
        "evalMetric"
    }:
        return False

    if not safe_int(trial["trialId"]):
        return False

    if trial["status"] not in {"SUCCEEDED", "FAILED"}:
        return False

    if not finite(trial["evalMetric"]):
        # NaN / infinity are invalid.
        return False

    return True


def selection_signature(body):
    return compact(body)


def save_selection(run_id, body, response, successful):
    sig = selection_signature(body)

    if run_id in RUNS:
        existing = RUNS[run_id]

        if existing["signature"] != sig:
            return jsonify({
                "error": "RUN_ID_CONFLICT"
            }), 409

        return jsonify(existing["response"])

    RUNS[run_id] = {
        "signature": sig,
        "response": response,
        "successful": successful,
    }

    return jsonify(response)


def select_phase(body):
    expected = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials",
    }

    if set(body.keys()) != expected:
        return invalid()

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return invalid()

    forbidden = body["forbiddenFeatures"]

    if not isinstance(forbidden, list):
        return invalid()

    if not all(isinstance(x, str) for x in forbidden):
        return invalid()

    limit = body["numTrialsLimit"]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        return invalid()

    rows = body["rows"]

    if not isinstance(rows, list) or not rows:
        return invalid()

    trials = body["trials"]

    if not isinstance(trials, list):
        return invalid()

    row_ids = []

    for row in rows:
        if not valid_select_row(row):
            return invalid()

        row_ids.append(row["id"])

    if len(row_ids) != len(set(row_ids)):
        return invalid()

    trial_ids = []

    for trial in trials:
        if not valid_trial(trial):
            return invalid()

        trial_ids.append(trial["trialId"])

    if len(trial_ids) != len(set(trial_ids)):
        return invalid()

    if len(trials) > limit:
        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["TRIAL_LIMIT_EXCEEDED"],
        }

        return save_selection(
            run_id,
            body,
            response,
            False
        )

    # Deduplicate by entity + UTC eventTime.
    groups = {}

    for row in rows:
        key = (
            row["entity"],
            utc_time(row["eventTime"])
        )

        groups.setdefault(key, []).append(row)

    retained = []

    for group in groups.values():
        group.sort(
            key=lambda row: (
                -row["version"],
                utf8(row["id"])
            )
        )
        retained.append(group[0])

    # Common feature names.
    common = None

    for row in retained:
        names = set(row["features"].keys())

        if common is None:
            common = names
        else:
            common &= names

    if common is None:
        common = set()

    forbidden_set = set(forbidden)

    feature_names = sorted(
        [
            name
            for name in common
            if name not in forbidden_set
        ],
        key=utf8
    )

    train_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ],
        key=utf8
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ],
        key=utf8
    )

    successful_trials = [
        trial
        for trial in trials
        if (
            trial["status"] == "SUCCEEDED"
            and finite(trial["evalMetric"])
        )
    ]

    if not successful_trials:
        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"],
        }

        return save_selection(
            run_id,
            body,
            response,
            False
        )

    # Highest metric; exact tie -> smallest trialId.
    successful_trials.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    selected_trial = successful_trials[0]["trialId"]

    dataset = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    digest = hashlib.sha256(
        compact(dataset).encode("utf-8")
    ).hexdigest()

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": [],
    }

    return save_selection(
        run_id,
        body,
        response,
        True
    )


# ============================================================
# EVALUATE
# ============================================================

def valid_test_row(row):
    if not isinstance(row, dict):
        return False

    if set(row.keys()) != {
        "label",
        "prediction",
        "slice"
    }:
        return False

    if (
        not isinstance(row["label"], int)
        or isinstance(row["label"], bool)
        or row["label"] not in {0, 1}
    ):
        return False

    if (
        not isinstance(row["prediction"], int)
        or isinstance(row["prediction"], bool)
        or row["prediction"] not in {0, 1}
    ):
        return False

    if (
        not isinstance(row["slice"], str)
        or not row["slice"]
    ):
        return False

    return True


def evaluate_phase(body):
    expected = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes",
    }

    if set(body.keys()) != expected:
        return invalid()

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return invalid()

    selected_trial = body["selectedTrialId"]

    if not safe_int(selected_trial):
        return invalid()

    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return invalid()

    metric_floor = body["metricFloor"]

    if (
        not finite(metric_floor)
        or not 0 <= float(metric_floor) <= 1
    ):
        return invalid()

    required_slices = body["requiredSlices"]

    if not isinstance(required_slices, dict):
        return invalid()

    for name, floor in required_slices.items():
        if not isinstance(name, str):
            return invalid()

        if (
            not finite(floor)
            or not 0 <= float(floor) <= 1
        ):
            return invalid()

    rows = body["rows"]

    if not isinstance(rows, list):
        return invalid()

    bytes_processed = body["bytesProcessed"]
    max_bytes = body["maxBytes"]

    if not safe_int(bytes_processed):
        return invalid()

    if not safe_int(max_bytes):
        return invalid()

    reason_codes = []

    stored = RUNS.get(run_id)

    lineage_ok = (
        stored is not None
        and stored["successful"]
        and stored["response"]["selectedTrialId"]
            == selected_trial
        and stored["response"]["datasetDigest"]
            == digest
    )

    if not lineage_ok:
        reason_codes.append("INVALID_LINEAGE")

    rows_valid = all(
        valid_test_row(row)
        for row in rows
    )

    if not rows_valid:
        reason_codes.append("INVALID_TEST_ROW")

    test_metric = None

    # Empty or invalid rows: no aggregate/slice evaluation.
    if rows and rows_valid:

        correct = sum(
            row["label"] == row["prediction"]
            for row in rows
        )

        test_metric = round(
            correct / len(rows),
            12
        )

        if test_metric < float(metric_floor):
            reason_codes.append("AGGREGATE_FLOOR")

        for slice_name, floor in required_slices.items():

            slice_rows = [
                row
                for row in rows
                if row["slice"] == slice_name
            ]

            if not slice_rows:
                reason_codes.append(
                    f"MISSING_SLICE:{slice_name}"
                )
                continue

            correct_slice = sum(
                row["label"] == row["prediction"]
                for row in slice_rows
            )

            accuracy = round(
                correct_slice / len(slice_rows),
                12
            )

            if accuracy < float(floor):
                reason_codes.append(
                    f"SLICE_FLOOR:{slice_name}"
                )

    if bytes_processed > max_bytes:
        reason_codes.append("BYTE_LIMIT")

    reason_codes = sort_codes(reason_codes)

    slice_failed = any(
        code.startswith("MISSING_SLICE:")
        or code.startswith("SLICE_FLOOR:")
        for code in reason_codes
    )

    critical_slice_pass = (
        lineage_ok
        and rows_valid
        and bool(rows)
        and not slice_failed
    )

    decision = (
        "admit"
        if not reason_codes
        else "reject"
    )

    return jsonify({
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes,
    })


# ============================================================
# Endpoint
# ============================================================

@app.route("/bqml", methods=["POST"])
def bqml():

    body = None

    if request.is_json:
        try:
            body = request.get_json()
        except Exception:
            body = None

    if not isinstance(body, dict):
        return invalid()

    phase = body.get("phase")

    if phase == "select":
        return select_phase(body)

    if phase == "evaluate":
        return evaluate_phase(body)

    return invalid()


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "bqml-experiment-gate",
        "status": "ok"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
