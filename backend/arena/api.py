import logging
from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from .engine import generate, prepare, fit_experiment
from .observation import fit_observation
from .timeline import record_timeline
from .parameters import PARAMETERS

blueprint = Blueprint("arena", __name__, url_prefix="/api/arena")


@blueprint.get("/parameters")
def parameters():
    return jsonify(status="success", models=PARAMETERS)


def respond(action):
    try:
        # Limit this module only; do not change upload limits of original labs.
        if request.content_length is not None and request.content_length > 180000:
            raise RequestEntityTooLarge()
        raw = request.stream.read(180001)
        if len(raw) > 180000:
            raise RequestEntityTooLarge()
        import json
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError("请求必须为有效 JSON。")
        if not isinstance(payload, dict):
            raise ValueError("请求必须为 JSON 对象。")
        return jsonify(status="success", **action(payload))
    except RequestEntityTooLarge:
        return jsonify(status="error", message="请求过大，请限制到 600 个样本。"), 413
    except (ValueError, TypeError, BadRequest) as exc:
        return jsonify(status="error", message=str(exc)), 400
    except Exception:
        logging.exception("Arena experiment failed")
        return jsonify(status="error", message="计算失败，请检查数据或降低模型复杂度后重试。"), 500


@blueprint.post("/generate")
def dataset():
    return respond(generate)


@blueprint.post("/prepare")
def preview():
    return respond(lambda p: prepare(p)[0])


@blueprint.post("/run")
def run():
    return respond(fit_experiment)


@blueprint.post("/observe")
def observe():
    return respond(fit_observation)


@blueprint.post("/timeline")
def timeline():
    return respond(record_timeline)
