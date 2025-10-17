# LIMINAL ProtoConsciousness

LIMINAL ProtoConsciousness is an experimental platform exploring emergent analytical behaviors and adaptive learning signals.
This repository contains the foundational modules outlined in the MVP specification to begin iterating on proto-conscious processing loops.

LIMINAL ProtoConsciousness — the living analytical seed that learns through reflection.

See the [MVP specification](docs/MVP_SPEC.md) for detailed requirements and roadmap guidance.

### Collect news (RSS → JSONL)
```bash
python src/collector.py --feeds configs/feeds.txt --out data/raw/news_$(date +%Y%m%d).jsonl --min-length 40 --max-items 1000
```
