import json

from cseval import cli
from cseval.aspects import SPECS
from cseval.llm import FakeLLM

from .conftest import DATA_DIR


class CountingFakeLLM(FakeLLM):
    hits = 0
    misses = 0


def test_score_then_evaluate(tmp_path, monkeypatch, capsys, dataset):
    by_pair = {(item.hatespeech, item.counterspeech): item for item in dataset.items}

    def respond(request):
        system, user = request.messages[0][1], request.messages[1][1]
        spec = next(s for s in SPECS.values() if s.criteria in system)
        hatespeech = user.split("Hate Speech: ")[1].split("\n")[0]
        item = by_pair[(hatespeech, user.split("Counterspeech: ")[1].split("\n")[0])]
        gold = item.gold[spec.aspect]
        return json.dumps({"score": spec.low if gold is None else round(gold)})

    monkeypatch.setattr(cli, "build_llm", lambda config: CountingFakeLLM(respond, model="fake"))
    config = tmp_path / "score.toml"
    config.write_text(
        f'[data]\ndir = "{DATA_DIR}"\nsplit = "test"\n\n'
        '[model]\nprovider = "openai"\nname = "fake"\n\n'
        '[judge]\nkind = "zero-shot"\n\n'
        f'[run]\ndir = "{tmp_path / "runs"}"\n',
        encoding="utf-8",
    )
    cli.main(["score", str(config), "--limit", "60", "--yes"])
    (run_dir,) = (tmp_path / "runs").iterdir()
    records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 60 * 4 and all(r["score"] is not None for r in records)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["data_fingerprint"] == dataset.fingerprint and manifest["model"] == "fake"

    capsys.readouterr()
    cli.main(["evaluate", str(run_dir), "--level", "dataset"])
    out = capsys.readouterr().out
    assert "| zero-shot · fake |" in out
    assert "relevance: n=60, unparsed=0" in out
    rel_rho = float(out.split("| zero-shot · fake | ")[1].split(" | ")[0])
    assert rel_rho > 0.9
