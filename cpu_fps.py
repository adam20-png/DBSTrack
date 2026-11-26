"""CPU speed test: runs test.py with the GPU hidden; results in results_cpu/.

    python cpu_fps.py OTB100
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch  # noqa: E402
import test as T  # noqa: E402

print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'threads:', torch.get_num_threads(), flush=True)
T.LABEL = 'cpu'
T.BENCHMARK = sys.argv[1] if len(sys.argv) > 1 else 'OTB100'
T.OUT_DIR = os.path.join(ROOT, 'results_cpu')
T.main()
