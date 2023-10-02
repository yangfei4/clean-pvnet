from lib.config import cfg, args
import numpy as np
import os

def run_rgb():
    import glob
    from scipy.misc import imread
    import matplotlib.pyplot as plt

    syn_ids = sorted(os.listdir('data/ShapeNet/renders/02958343/'))[-10:]
    for syn_id in syn_ids:
        pkl_paths = glob.glob('data/ShapeNet/renders/02958343/{}/*.pkl'.format(syn_id))
        np.random.shuffle(pkl_paths)
        for pkl_path in pkl_paths:
            img_path = pkl_path.replace('_RT.pkl', '.png')
            img = imread(img_path)
            plt.imshow(img)
            plt.show()


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass

def run_inference():
    import torch
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from PIL import Image
    from lib.visualizers import make_visualizer
    from lib.datasets.transforms import make_transforms
    from lib.datasets import make_data_loader

    # image_path = "/pvnet/data/FIT/image_basler_5k_128x128_04.png"
    # image_path = "/pvnet/data/FIT/mainshell_04.png"
    # image_path = "/pvnet/data/FIT/topshell_04.png"
    # image_path = "/pvnet/data/FIT/insert_mold_2107_1323.png"

    # lagest error
    # image_path = "/pvnet/data/evaluation/insert_mold_487.png"
    # image_path = "/pvnet/data/evaluation/mainshell_1421.png"
    # image_path = "/pvnet/data/evaluation/topshell_1984.png"

    # image_path = "/pvnet/data/FIT/test_crop/1-dim128x128_u1921_v2171_two_parts.png"
    # image_path = "/pvnet/data/FIT/test_crop/2-dim128x128_u1792_v2397_two_parts.png"
    # image_path = "/pvnet/data/FIT/test_crop/3-dim128x128_u2154_v2089_two_parts.png"
    # image_path = "/pvnet/data/FIT/test_crop/4-dim128x128_u1963_v2171_two_parts.png"

    # image_path = "/pvnet/data/FIT/test_crop/1-dim128x128_u2243_v2089_truncated.png"
    # image_path = "/pvnet/data/FIT/test_crop/2-dim128x128_u1888_v1897_truncated.png"
    # image_path = "/pvnet/data/FIT/test_crop/3-dim128x128_u2011_v2205_truncated.png"
    # image_path = "/pvnet/data/FIT/test_crop/4-dim128x128_u1580_v2232_truncated.png"

    image_path = "/pvnet/data/FIT/insert_mold_640x480.png"
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    data_loader = make_data_loader(cfg, is_train=False)
    
    batch_example = None # will be used to load kpts annotation
    for batch in data_loader:
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        batch_example = batch
        break

    network.eval()

    image = Image.open(image_path).convert('RGB')
    # Preprocess the image

    transform = make_transforms(cfg, is_train=False)
    processed_image, _, _ = transform(image)
    processed_image = np.array(processed_image).astype(np.float32)


    # Convert the preprocessed image to a tensor and move it to GPU
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).cuda().float()

    with torch.no_grad():
        output = network(input_tensor)

    visualizer = make_visualizer(cfg)
    # fig = visualizer.visualize_output(image, output)
    visualizer.visualize_output(image, output, batch_example)



def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'], batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(f"Inference time on {len(data_loader)} data: ", total_time)
    print(f"Inference time on single data: ", total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    torch.manual_seed(0)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    # import pdb;pdb.set_trace()
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

def run_visualize_gt():
    import torch
    import tqdm

    from lib.config import cfg
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        visualizer.visualize_gt(batch)

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        fig = visualizer.visualize(output, batch)

        # Save the figure
        save_dir = "/pvnet/data/visualization"
        img_id = int(batch['img_id'][0])
        save_path = os.path.join(save_dir, f'{img_id}.png')
        fig.savefig(save_path)


def run_visualize_train():
    # Visualize data annotation -- mask, keypoints
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    # network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize_train(output, batch)


def run_analyze():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.analyzers import make_analyzer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    analyzer = make_analyzer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        analyzer.analyze(output, batch)


def run_net_utils():
    from lib.utils import net_utils
    import torch
    import os

    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    pretrained_model = torch.load(model_path)
    net = pretrained_model['net']
    net = net_utils.remove_net_prefix(net, 'dla.')
    net = net_utils.remove_net_prefix(net, 'cp.')
    pretrained_model['net'] = net
    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    os.system('mkdir -p {}'.format(os.path.dirname(model_path)))
    torch.save(pretrained_model, model_path)


def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


def run_ycb():
    from lib.datasets.ycb import handle_ycb
    handle_ycb.collect_ycb()


def run_render():
    from lib.utils.renderer import opengl_utils
    from lib.utils.vsd import inout
    from lib.utils.linemod import linemod_config
    import matplotlib.pyplot as plt

    obj_path = 'data/linemod/cat/cat.ply'
    model = inout.load_ply(obj_path)
    model['pts'] = model['pts'] * 1000.
    im_size = (640, 300)
    opengl = opengl_utils.NormalRender(model, im_size)

    K = linemod_config.linemod_K
    pose = np.load('data/linemod/cat/pose/pose0.npy')
    depth = opengl.render(im_size, 100, 10000, K, pose[:, :3], pose[:, 3:] * 1000)

    plt.imshow(depth)
    plt.show()

def run_insert_mold():
    from tools import handle_custom_dataset
    data_root = 'data/insert_mold_train'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_insert_mold_test():
    from tools import handle_custom_dataset
    data_root = 'data/insert_mold_test'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_mainshell():
    from tools import handle_custom_dataset
    data_root = 'data/mainshell_train'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_mainshell_test():
    from tools import handle_custom_dataset
    data_root = 'data/mainshell_test'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_topshell():
    from tools import handle_custom_dataset
    data_root = 'data/topshell_train'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_topshell_test():
    from tools import handle_custom_dataset
    data_root = 'data/topshell_test'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_custom():
    from tools import handle_custom_dataset
    data_root = 'data/custom'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_custom_test():
    from tools import handle_custom_dataset
    data_root = 'data/custom_test'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)

def run_detector_pvnet():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)

def run_demo():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image

    torch.manual_seed(0)
    meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    demo_images = glob.glob(cfg.demo_path + '/*jpg')

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for demo_image in demo_images:
        demo_image = np.array(Image.open(demo_image)).astype(np.float32)
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        visualizer.visualize_demo(output, inp, meta)

if __name__ == '__main__':
    globals()['run_'+args.type]()