import os

root_path = os.path.dirname(os.path.realpath(__file__))
mri_3d_path = '/home/lxs/ADNI/npy'
mri_2d_path = os.path.join(root_path, 'datasets/2d_mri')
if not os.path.exists(mri_2d_path):
    os.mkdir(mri_2d_path)

label_map = {
	'CN': 0,
	'AD': 1
}
