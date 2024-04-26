from .llff import LLFFDataset
from .blender import BlenderDataset
from .realdata import RealdataDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .unbounded import UnboundedDataset
from .dtu import DtuDataset
from .scannet import ScanNetDataset

dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'own_data':YourOwnDataset,
                'unbounded':UnboundedDataset,
                'realdata': RealdataDataset,
                'dtu': DtuDataset,
                'scannet': ScanNetDataset}
