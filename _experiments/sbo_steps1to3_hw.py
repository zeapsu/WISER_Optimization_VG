from pathlib import Path

ROOT = Path(__file__).parent.parent
print(ROOT)

import sys
sys.path.append(str(ROOT))
from sbo_steps1to3 import execute_multiple_runs, warm_start_nft

from doe import doe

if __name__ == '__main__':
    execute_multiple_runs(**doe['1/109bonds/TwoLocal2rep_bilinear_piby3_fez_0.1'], instance='vanguard/internal/default', run_on_serverless=False)
    # warm_start_nft(f'{ROOT}/data/1/109bonds/bfcdR2rep_color_piby3_marrakesh_0.1/exp0.pkl', 0, 219, instance='vanguard/internal/default', max_epoch=2)
    