```python
import hashlib
import json
import math
import re
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, request

app = Flask(__name__)

SAFE_MAX = 9007199254740991
RUNS = {}

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def b(value):
    return value.encode("utf-8")


def codes_sorted(values):
    return sorted(set(values), key=b)


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_MAX
    )


def finite_number(value):
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def invalid_input():
    return jsonify({"error": "INVALID_INPUT"}), 400


# ------------------------------------------------------------
# Timestamp parsing
# ------------------------------------------------------------

def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))
    second = int(match.group(6))

    fraction = match.group(7)
    offset = match.group(8)

    microsecond = 0

    if fraction is not None:
        microsecond = int(
            fraction.ljust(3, "0")
        ) * 1000

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

        hours = int(offset[1:3])
        minutes = int(offset[4:6])

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


def utc_timestamp(value):
    dt = parse_timestamp(value)

    if dt is None:
        return None

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


# ------------------------------------------------------------
# Selection-row validation
# ------------------------------------------------------------

def validate_feature(feature):
    if not isinstance(feature, dict):
        return False

    if set(feature.keys()) != {
        "value",
        "availableAt"
    }:
        return False

    # Feature values are DATA. Do not interpret their contents.
    if not isinstance(feature["value"], str):
        return False

    if parse_timestamp(feature["availableAt"]) is None:
        return False

    return True


def validate_selection_row(row):
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

    if parse_timestamp(row["eventTime"]) is None:
        return False

    if parse_timestamp(row["predictionTime"]) is None:
        return False

    if not safe_integer(row["version"]):
        return False

    if row["split"] not in {"TRAIN", "EVAL"}:
        return False

    if not isinstance(row["features"], dict):
        return False

    for name, feature in row["features"].items():

        if not isinstance(name, str):
            return False

        if not validate_feature(feature):
            return False

    return True


# ------------------------------------------------------------
# Trial validation
# ------------------------------------------------------------

def validate_trial(trial):
    if not isinstance(trial, dict):
        return False

    if set(trial.keys()) != {
        "trialId",
        "status",
        "evalMetric"
    }:
        return False

    if not safe_integer(trial["trialId"]):
        return False

    if trial["status"] not in {
        "SUCCEEDED",
        "FAILED"
    }:
        return False

    # Metric must be numeric. NaN / infinity are allowed to exist
    # syntactically but are not eligible successful trials.
    if isinstance(trial["evalMetric"], bool):
        return False

    if not isinstance(
        trial["evalMetric"],
        (int, float)
    ):
        return False

    return True


# ------------------------------------------------------------
# Select phase
# ------------------------------------------------------------

def select_phase(body):

    expected = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials"
    }

    if set(body.keys()) != expected:
        return invalid_input()

    if body["phase"] != "select":
        return invalid_input()

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return invalid_input()

    forbidden = body["forbiddenFeatures"]

    if not isinstance(forbidden, list):
        return invalid_input()

    if not all(
        isinstance(x, str)
        for x in forbidden
    ):
        return invalid_input()

    trial_limit = body["numTrialsLimit"]

    if (
        not isinstance(trial_limit, int)
        or isinstance(trial_limit, bool)
        or trial_limit <= 0
    ):
        return invalid_input()

    rows = body["rows"]

    if not isinstance(rows, list) or not rows:
        return invalid_input()

    trials = body["trials"]

    if not isinstance(trials, list):
        return invalid_input()

    # Validate rows.
    ids = []

    for row in rows:
        if not validate_selection_row(row):
            return invalid_input()

        ids.append(row["id"])

    if len(ids) != len(set(ids)):
        return invalid_input()

    # Validate trials.
    trial_ids = []

    for trial in trials:
        if not validate_trial(trial):
            return invalid_input()

        trial_ids.append(trial["trialId"])

    if len(trial_ids) != len(set(trial_ids)):
        return invalid_input()

    # --------------------------------------------------------
    # Contract trial limit
    # --------------------------------------------------------

    if len(trials) > trial_limit:

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

        return save_selection(
            run_id,
            body,
            response,
            False
        )

    # --------------------------------------------------------
    # Dedup:
    # [entity, UTC(eventTime)]
    #
    # Highest version wins.
    # Tie -> UTF-8 smallest ID.
    # --------------------------------------------------------

    groups = {}

    for row in rows:

        key = (
            row["entity"],
            utc_timestamp(row["eventTime"])
        )

        groups.setdefault(key, []).append(row)

    retained = []

    for group in groups.values():

        group.sort(
            key=lambda row: (
                -row["version"],
                b(row["id"])
            )
        )

        retained.append(group[0])

    # --------------------------------------------------------
    # Point-in-time feature eligibility
    #
    # A feature is eligible only when:
    # 1. it appears in EVERY retained row
    # 2. it isn't forbidden
    # 3. for EVERY retained row:
    #       availableAt <= predictionTime
    #
    # Features failing the availability test are excluded.
    # --------------------------------------------------------

    common_features = None

    for row in retained:

        names = set(
            row["features"].keys()
        )

        if common_features is None:
            common_features = names
        else:
            common_features &= names

    if common_features is None:
        common_features = set()

    forbidden_set = set(forbidden)

    eligible_features = []

    for name in common_features:

        if name in forbidden_set:
            continue

        available_everywhere = True

        for row in retained:

            feature = row["features"][name]

            available_at = parse_timestamp(
                feature["availableAt"]
            )

            prediction_time = parse_timestamp(
                row["predictionTime"]
            )

            if available_at > prediction_time:
                available_everywhere = False
                break

        if available_everywhere:
            eligible_features.append(name)

    feature_names = sorted(
        eligible_features,
        key=b
    )

    # --------------------------------------------------------
    # Split IDs
    # --------------------------------------------------------

    train_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ],
        key=b
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ],
        key=b
    )

    # --------------------------------------------------------
    # Trial selection
    #
    # Only finite SUCCEEDED trials.
    # Highest metric wins.
    # Exact metric tie -> smallest trialId.
    # --------------------------------------------------------

    eligible_trials = [
        trial
        for trial in trials
        if (
            trial["status"] == "SUCCEEDED"
            and finite_number(trial["evalMetric"])
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

        return save_selection(
            run_id,
            body,
            response,
            False
        )

    eligible_trials.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"]
        )
    )

    selected_trial = eligible_trials[0][
        "trialId"
    ]

    # --------------------------------------------------------
    # Frozen dataset digest
    # --------------------------------------------------------

    dataset = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    dataset_digest = hashlib.sha256(
        compact(dataset).encode("utf-8")
    ).hexdigest()

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": []
    }

    return save_selection(
        run_id,
        body,
        response,
        True
    )


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

def save_selection(
    run_id,
    body,
    response,
    successful
):
    signature = compact(body)

    if run_id in RUNS:

        previous = RUNS[run_id]

        if previous["signature"] != signature:
            return (
                jsonify({
                    "error": "RUN_ID_CONFLICT"
                }),
                409
            )

        return jsonify(
            previous["response"]
        )

    RUNS[run_id] = {
        "signature": signature,
        "response": response,
        "successful": successful
    }

    return jsonify(response)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def validate_test_row(row):

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
        "maxBytes"
    }

    if set(body.keys()) != expected:
        return invalid_input()

    if body["phase"] != "evaluate":
        return invalid_input()

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return invalid_input()

    selected_trial = body[
        "selectedTrialId"
    ]

    if not safe_integer(selected_trial):
        return invalid_input()

    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return invalid_input()

    metric_floor = body["metricFloor"]

    if (
        not finite_number(metric_floor)
        or not 0 <= float(metric_floor) <= 1
    ):
        return invalid_input()

    required_slices = body[
        "requiredSlices"
    ]

    if not isinstance(required_slices, dict):
        return invalid_input()

    for name, floor in required_slices.items():

        if not isinstance(name, str):
            return invalid_input()

        if (
            not finite_number(floor)
            or not 0 <= float(floor) <= 1
        ):
            return invalid_input()

    rows = body["rows"]

    if not isinstance(rows, list):
        return invalid_input()

    bytes_processed = body[
        "bytesProcessed"
    ]

    max_bytes = body["maxBytes"]

    if not safe_integer(bytes_processed):
        return invalid_input()

    if not safe_integer(max_bytes):
        return invalid_input()

    reasons = []

    # --------------------------------------------------------
    # Frozen lineage
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
        reasons.append(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # Test row validation
    # --------------------------------------------------------

    rows_valid = all(
        validate_test_row(row)
        for row in rows
    )

    if not rows_valid:
        reasons.append(
            "INVALID_TEST_ROW"
        )

    test_metric = None

    # Empty OR invalid test rows:
    # metric stays null and aggregate/slice gates
    # are skipped.
    if rows and rows_valid:

        correct = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = round(
            correct / len(rows),
            12
        )

        if test_metric < float(metric_floor):
            reasons.append(
                "AGGREGATE_FLOOR"
            )

        # Required slices.
        for slice_name, floor in required_slices.items():

            slice_rows = [
                row
                for row in rows
                if row["slice"]
                == slice_name
            ]

            if not slice_rows:

                reasons.append(
                    f"MISSING_SLICE:{slice_name}"
                )

                continue

            slice_correct = sum(
                1
                for row in slice_rows
                if row["label"]
                == row["prediction"]
            )

            slice_accuracy = round(
                slice_correct
                / len(slice_rows),
                12
            )

            if slice_accuracy < float(floor):

                reasons.append(
                    f"SLICE_FLOOR:{slice_name}"
                )

    # --------------------------------------------------------
    # Cost gate
    # --------------------------------------------------------

    if bytes_processed > max_bytes:
        reasons.append(
            "BYTE_LIMIT"
        )

    reasons = codes_sorted(reasons)

    slice_failed = any(
        code.startswith("MISSING_SLICE:")
        or code.startswith("SLICE_FLOOR:")
        for code in reasons
    )

    critical_slice_pass = (
        lineage_ok
        and rows_valid
        and bool(rows)
        and not slice_failed
    )

    decision = (
        "admit"
        if not reasons
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


# ------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------

@app.route("/bqml", methods=["POST"])
def bqml():

    if not request.is_json:
        return invalid_input()

    try:
        body = request.get_json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    phase = body.get("phase")

    if phase == "select":
        return select_phase(body)

    if phase == "evaluate":
        return evaluate_phase(body)

    return invalid_input()


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
```
