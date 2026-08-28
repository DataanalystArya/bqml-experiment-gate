```python
import hashlib
import json
import math
import re
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, request

app = Flask(__name__)

RUNS = {}
SAFE_MAX = 9007199254740991

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utf8(x):
    return x.encode("utf-8")


def compact(x):
    return json.dumps(
        x,
        ensure_ascii=False,
        separators=(",", ":")
    )


def sorted_codes(codes):
    return sorted(set(codes), key=utf8)


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_MAX
    )


def finite(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


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
    zone = m.group(8)

    if frac is None:
        micro = 0
    else:
        micro = int(frac.ljust(3, "0")) * 1000

    try:
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            micro
        )
    except ValueError:
        return None

    if zone == "Z":
        tz = timezone.utc
    else:
        sign = 1 if zone[0] == "+" else -1
        hours = int(zone[1:3])
        minutes = int(zone[4:6])

        if hours > 14:
            return None

        if minutes > 59:
            return None

        if hours == 14 and minutes != 0:
            return None

        tz = timezone(
            sign * timedelta(
                hours=hours,
                minutes=minutes
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


def bad_input():
    return jsonify({"error": "INVALID_INPUT"}), 400


# ============================================================
# SELECT VALIDATION
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

    if set(row.keys()) != {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    }:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if parse_time(row["eventTime"]) is None:
        return False

    if parse_time(row["predictionTime"]) is None:
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

    if trial["status"] not in {
        "SUCCEEDED",
        "FAILED"
    }:
        return False

    if not isinstance(
        trial["evalMetric"],
        (int, float)
    ) or isinstance(
        trial["evalMetric"],
        bool
    ):
        return False

    return True


# ============================================================
# STATE
# ============================================================

def save_run(run_id, body, response, successful):
    signature = compact(body)

    if run_id in RUNS:
        old = RUNS[run_id]

        if old["signature"] != signature:
            return jsonify({
                "error": "RUN_ID_CONFLICT"
            }), 409

        return jsonify(old["response"])

    RUNS[run_id] = {
        "signature": signature,
        "response": response,
        "successful": successful
    }

    return jsonify(response)


# ============================================================
# SELECT
# ============================================================

def select(body):

    required = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials"
    }

    if set(body.keys()) != required:
        return bad_input()

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return bad_input()

    forbidden = body["forbiddenFeatures"]

    if (
        not isinstance(forbidden, list)
        or not all(
            isinstance(x, str)
            for x in forbidden
        )
    ):
        return bad_input()

    limit = body["numTrialsLimit"]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        return bad_input()

    rows = body["rows"]

    if not isinstance(rows, list) or len(rows) == 0:
        return bad_input()

    trials = body["trials"]

    if not isinstance(trials, list):
        return bad_input()

    # --------------------------------------------------------
    # Validate every selection row.
    # --------------------------------------------------------

    row_ids = []

    for row in rows:
        if not valid_select_row(row):
            return bad_input()

        row_ids.append(row["id"])

    if len(row_ids) != len(set(row_ids)):
        return bad_input()

    # --------------------------------------------------------
    # Validate trials.
    # --------------------------------------------------------

    trial_ids = []

    for trial in trials:
        if not valid_trial(trial):
            return bad_input()

        trial_ids.append(trial["trialId"])

    if len(trial_ids) != len(set(trial_ids)):
        return bad_input()

    # --------------------------------------------------------
    # Trial count limit.
    # --------------------------------------------------------

    if len(trials) > limit:

        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": [
                "TRIAL_LIMIT_EXCEEDED"
            ]
        }

        return save_run(
            run_id,
            body,
            response,
            False
        )

    # --------------------------------------------------------
    # Deduplicate:
    # [entity, UTC(eventTime)]
    #
    # Highest version wins.
    # Exact version tie -> smallest UTF-8 ID.
    # --------------------------------------------------------

    groups = {}

    for row in rows:

        key = (
            row["entity"],
            utc_time(row["eventTime"])
        )

        groups.setdefault(key, []).append(row)

    retained = []

    for group in groups.values():

        winner = min(
            group,
            key=lambda r: (
                -r["version"],
                utf8(r["id"])
            )
        )

        retained.append(winner)

    # --------------------------------------------------------
    # Shared feature names.
    # --------------------------------------------------------

    shared = None

    for row in retained:

        names = set(
            row["features"].keys()
        )

        if shared is None:
            shared = names
        else:
            shared &= names

    if shared is None:
        shared = set()

    forbidden_set = set(forbidden)

    # --------------------------------------------------------
    # Point-in-time eligibility.
    #
    # Feature must:
    # - exist in every retained row
    # - not be forbidden
    # - be available by predictionTime in EVERY row
    # --------------------------------------------------------

    eligible = set()

    for name in shared:

        if name in forbidden_set:
            continue

        ok = True

        for row in retained:

            feature_time = parse_time(
                row["features"][name]["availableAt"]
            )

            prediction_time = parse_time(
                row["predictionTime"]
            )

            if feature_time > prediction_time:
                ok = False
                break

        if ok:
            eligible.add(name)

    feature_names = sorted(
        eligible,
        key=utf8
    )

    # --------------------------------------------------------
    # TRAIN / EVAL IDs.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Trial eligibility.
    # Only SUCCEEDED + finite.
    # --------------------------------------------------------

    eligible_trials = [
        t for t in trials
        if (
            t["status"] == "SUCCEEDED"
            and finite(t["evalMetric"])
        )
    ]

    if not eligible_trials:

        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": [
                "NO_SUCCESSFUL_TRIAL"
            ]
        }

        return save_run(
            run_id,
            body,
            response,
            False
        )

    # Highest metric first.
    # Exact tie -> smallest integer trialId.
    eligible_trials.sort(
        key=lambda t: (
            -float(t["evalMetric"]),
            t["trialId"]
        )
    )

    selected_trial = eligible_trials[0]["trialId"]

    # --------------------------------------------------------
    # Exact datasetDigest shape and order.
    # --------------------------------------------------------

    dataset = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
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
        "reasonCodes": []
    }

    return save_run(
        run_id,
        body,
        response,
        True
    )


# ============================================================
# EVALUATION VALIDATION
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


# ============================================================
# EVALUATE
# ============================================================

def evaluate(body):

    required = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes"
    }

    if set(body.keys()) != required:
        return bad_input()

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return bad_input()

    selected_trial = body["selectedTrialId"]

    if not safe_int(selected_trial):
        return bad_input()

    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or HEX64.fullmatch(digest) is None
    ):
        return bad_input()

    metric_floor = body["metricFloor"]

    if (
        not finite(metric_floor)
        or float(metric_floor) < 0
        or float(metric_floor) > 1
    ):
        return bad_input()

    required_slices = body["requiredSlices"]

    if not isinstance(required_slices, dict):
        return bad_input()

    for name, floor in required_slices.items():

        if not isinstance(name, str):
            return bad_input()

        if (
            not finite(floor)
            or float(floor) < 0
            or float(floor) > 1
        ):
            return bad_input()

    rows = body["rows"]

    if not isinstance(rows, list):
        return bad_input()

    bytes_processed = body["bytesProcessed"]
    max_bytes = body["maxBytes"]

    if not safe_int(bytes_processed):
        return bad_input()

    if not safe_int(max_bytes):
        return bad_input()

    reasons = []

    # --------------------------------------------------------
    # Frozen lineage.
    # --------------------------------------------------------

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
        reasons.append("INVALID_LINEAGE")

    # --------------------------------------------------------
    # Validate ALL test rows.
    # --------------------------------------------------------

    rows_valid = all(
        valid_test_row(row)
        for row in rows
    )

    if not rows_valid:
        reasons.append("INVALID_TEST_ROW")

    test_metric = None

    # Empty OR invalid rows:
    # metric stays null.
    # aggregate/slice gates are skipped.
    if rows and rows_valid:

        correct = sum(
            1
            for row in rows
            if row["label"] == row["prediction"]
        )

        test_metric = round(
            correct / len(rows),
            12
        )

        if test_metric < float(metric_floor):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # Required slices.
        # ----------------------------------------------------

        for slice_name, floor in required_slices.items():

            matching = [
                row
                for row in rows
                if row["slice"] == slice_name
            ]

            if not matching:

                reasons.append(
                    f"MISSING_SLICE:{slice_name}"
                )

                continue

            correct_slice = sum(
                1
                for row in matching
                if row["label"]
                == row["prediction"]
            )

            accuracy = round(
                correct_slice / len(matching),
                12
            )

            if accuracy < float(floor):

                reasons.append(
                    f"SLICE_FLOOR:{slice_name}"
                )

    # --------------------------------------------------------
    # Byte gate.
    # --------------------------------------------------------

    if bytes_processed > max_bytes:
        reasons.append("BYTE_LIMIT")

    reasons = sorted_codes(reasons)

    slice_failed = any(
        x.startswith("MISSING_SLICE:")
        or x.startswith("SLICE_FLOOR:")
        for x in reasons
    )

    critical_slice_pass = (
        lineage_ok
        and rows_valid
        and bool(rows)
        and not slice_failed
    )

    decision = (
        "admit"
        if len(reasons) == 0
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
        "reasonCodes": reasons
    })


# ============================================================
# HTTP
# ============================================================

@app.route("/bqml", methods=["POST"])
def bqml():

    if not request.is_json:
        return bad_input()

    try:
        body = request.get_json()
    except Exception:
        return bad_input()

    if not isinstance(body, dict):
        return bad_input()

    phase = body.get("phase")

    if phase == "select":
        return select(body)

    if phase == "evaluate":
        return evaluate(body)

    return bad_input()


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "bqml-experiment-gate",
        "status": "ok"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
```
