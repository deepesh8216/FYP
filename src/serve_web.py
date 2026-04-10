"""
Week 8: start the API from project root.

  python3 src/serve_web.py

Optional env: FAKE_NEWS_CHECKPOINT=path/to/best_model.pt
  (Week 5: outputs/week5/attention/seed_*/best_model.pt
   Week 6: outputs/week6/finetune_attention/seed_*/best_model.pt)
"""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
