
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
import time
import warnings

import cupy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

#from cupyx.scipy.ndimage import distance_transform_edt

#from convexAdam.convex_adam_utils import (MINDSSC, correlate, coupled_convex,
#                                          inverse_consistency)

warnings.filterwarnings("ignore")
def sort_rank(value):
    rank1 = torch.ones_like(value)
    rank1[value.sort().indices] = torch.linspace(1,.1,len(value)).to(value.device)
    return rank1
def cupy_hd95(fixed, moving, num_labels, precision=1):
    fixed_ = F.interpolate(F.one_hot(fixed, num_labels+1).permute(3, 0, 1, 2).unsqueeze(0).to(torch.uint8)[:, 1:], scale_factor=precision)[0]
    moving_ = F.interpolate(F.one_hot(moving, num_labels+1).permute(3, 0, 1, 2).unsqueeze(0).to(torch.uint8)[:, 1:], scale_factor=precision)[0]
    fixed_c = cupy.asarray(fixed_)
    moving_c = cupy.asarray(moving_)
    hd95 = cupy.zeros(num_labels)
    for i in range(num_labels):
        if((fixed_[i].sum()>0)&(moving_[i].sum()>0)):
            dist1 = distance_transform_edt(fixed_c[i], float64_distances=False)
            surf1 = dist1 == 1
            dist1 += distance_transform_edt(1-fixed_c[i], float64_distances=False)
            
            dist2 = distance_transform_edt(moving_c[i], float64_distances=False)
            surf2 = dist2 == 1
            dist2 += distance_transform_edt(1-moving_c[i], float64_distances=False)

            hd95[i] = (np.maximum(np.percentile(dist1[surf2], 95), np.percentile(dist2[surf1], 95)))
        else:
            hd95[i] = 30#np.nan
    return torch.as_tensor(hd95) * 1/precision

def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


# process nnUNet features
def extract_features_nnunet(pred_fixed,
                    pred_moving,mult=10):

    eps=1e-32
    H,W,D = pred_fixed.shape[-3:]
    
    maxlabel = max(pred_fixed.long().reshape(-1).max(),pred_moving.long().reshape(-1).max())+1
    combined_bins = torch.bincount(pred_fixed.long().reshape(-1),minlength=maxlabel)+torch.bincount(pred_moving.long().reshape(-1),minlength=maxlabel)
    pos = torch.nonzero(combined_bins).reshape(-1)
    
    pred_fixed = F.one_hot(pred_fixed.cuda().view(1,H,W,D).long(),int(maxlabel))[:,:,:,:,pos]
    pred_moving = F.one_hot(pred_moving.cuda().view(1,H,W,D).long(),int(maxlabel))[:,:,:,:,pos]
    
    weight = 1/(torch.bincount(torch.cat((pred_fixed.permute(0,4,1,2,3).argmax(1).long().reshape(-1),pred_moving.permute(0,4,1,2,3).argmax(1).long().reshape(-1)),0))+eps).float().pow(.3)
    weight /= weight.mean()

    features_fix = mult*(pred_fixed.data.float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()
    features_mov = mult*(pred_moving.data.float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()

    return features_fix, features_mov


def jacobian_determinant_3d(dense_flow,convert1=True):
    B,_,H,W,D = dense_flow.size()
    if(convert1):
        dense_pix = dense_flow*(torch.Tensor([H-1,W-1,D-1])/2).view(1,3,1,1,1).to(dense_flow.device)
    else:
        dense_pix = dense_flow
    gradz = nn.Conv3d(3,3,(3,1,1),padding=(1,0,0),bias=False,groups=3)
    gradz.weight.data[:,0,:,0,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3,3,(1,3,1),padding=(0,1,0),bias=False,groups=3)
    grady.weight.data[:,0,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3,3,(1,1,3),padding=(0,0,1),bias=False,groups=3)
    gradx.weight.data[:,0,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradx.to(dense_flow.device)
    with torch.no_grad():
        jacobian = torch.cat((gradz(dense_pix),grady(dense_pix),gradx(dense_pix)),0)+torch.eye(3,3).view(3,3,1,1,1).to(dense_flow.device)
        jacobian = jacobian[:,:,2:-2,2:-2,2:-2]
        jac_det = jacobian[0,0,:,:,:]*(jacobian[1,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[1,2,:,:,:]*jacobian[2,1,:,:,:])-\
        jacobian[1,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[2,1,:,:,:])+\
        jacobian[2,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[1,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[1,1,:,:,:])

    return jac_det
def extract_features(
    img_fixed: torch.Tensor,
    img_moving: torch.Tensor,
    mind_r: int,
    mind_d: int,
    use_mask: bool,
    mask_fixed: torch.Tensor,
    mask_moving: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """Extract MIND and/or semantic nnUNet features"""

    # MIND features
    if use_mask:
        H,W,D = img_fixed.shape[-3:]

        #replicate masking
        avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
        avg3.cuda()
        
        mask = (avg3(mask_fixed.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        fixed_r = F.interpolate((img_fixed[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        fixed_r.view(-1)[mask.view(-1)!=0] = img_fixed.cuda().reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        moving_r = F.interpolate((img_moving[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        moving_r.view(-1)[mask.view(-1)!=0] = img_moving.cuda().reshape(-1)[mask.view(-1)!=0]

        features_fix = MINDSSC(fixed_r.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(moving_r.cuda(),mind_r,mind_d).half()
    else:
        img_fixed = img_fixed.unsqueeze(0).unsqueeze(0)
        img_moving = img_moving.unsqueeze(0).unsqueeze(0)
        features_fix = MINDSSC(img_fixed.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(img_moving.cuda(),mind_r,mind_d).half()
    
    return features_fix, features_mov

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist
#s 89 0.416 0.218 jstd tensor(0.0812)
#100%|████████████████████████████████████████████████████████████████████████████████████| 100/100 [10:33<00:00,  6.34s/it]
#tensor(19)
#tensor([0.3338, 0.1416]) tensor([0.0665, 0.0000]) tensor(2.7740)
#tensor([2.5000, 5.0000, 6.0000])

def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    
    # kernel size
    kernel_size = radius * 2 + 1
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    
    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    
    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    
    return mind


#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp,shape, ch=12):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

    torch.cuda.synchronize()
    t0 = time.time()
    #with torch.no_grad():
    #    mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw)).squeeze(0),disp_hw*2+1)
    #    mind_unfold = mind_unfold.view(ch,-1,(disp_hw*2+1)**2,W//grid_sp,D//grid_sp)
        

    ssd = torch.zeros((disp_hw*2+1),(disp_hw*2+1),(disp_hw*2+1),H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
    #ssd_argmin = torch.zeros(H//grid_sp,W//grid_sp,D//grid_sp).long()
    with torch.no_grad():
        ssd = torch.zeros((disp_hw*2+1),(disp_hw*2+1),(disp_hw*2+1),H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
        mind_mov_ = F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw))
        with torch.no_grad():
            for i in range(disp_hw*2+1):
                for j in range(disp_hw*2+1):
                    for k in range(disp_hw*2+1):
                        mind_sum = (mind_fix-mind_mov_[:,:,i:i+H//grid_sp,j:j+W//grid_sp,k:k+D//grid_sp]).pow(2).sum(1,keepdim=True)
                        ssd[k,j,i] = F.avg_pool3d(F.avg_pool3d(mind_sum,3,stride=1,padding=1),3,stride=1,padding=1).squeeze(1)
        ssd = ssd.view((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        #for i in range(disp_hw*2+1):
        #    mind_sum = (mind_fix.permute(1,2,0,3,4)-mind_unfold[:,i:i+H//grid_sp]).pow(2).sum(0,keepdim=True)
        #    ssd[i::(disp_hw*2+1)] = F.avg_pool3d(F.avg_pool3d(mind_sum.transpose(2,1),3,stride=1,padding=1),3,stride=1,padding=1).squeeze(1)
        #ssd = ssd.view(disp_hw*2+1,disp_hw*2+1,disp_hw*2+1,H//grid_sp,W//grid_sp,D//grid_sp).transpose(1,0).reshape((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        ssd_argmin = torch.argmin(ssd,0)#
    torch.cuda.synchronize()

    t1 = time.time()
    #print(t1-t0,'sec (ssd)')
    #gpu_usage()
    return ssd,ssd_argmin



#solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,shape):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

    disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)

    coeffs = torch.tensor([0.003,0.01,0.03,0.1,0.3,1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H//grid_sp):

                coupled = ssd[:,i,:,:]+coeffs[j]*(disp_mesh_t-disp_soft[:,:,i].view(3,1,-1)).pow(2).sum(0).view(-1,W//grid_sp,D//grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled,0)

        disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_coupled_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)

    return disp_soft



#enforce inverse consistency of forward and backward transform
def inverse_consistency(disp_field1s,disp_field2s,iter=20):
    B,C,H,W,D = disp_field1s.size()
    #make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3).to(disp_field1s.device).to(disp_field1s.dtype)
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5*(disp_field1s-F.grid_sample(disp_field2s,(identity+disp_field1s).permute(0,2,3,4,1)))
            disp_field2i = 0.5*(disp_field2s-F.grid_sample(disp_field1s,(identity+disp_field2s).permute(0,2,3,4,1)))

    return disp_field1i,disp_field2i



def combineDeformation3d(disp_1st,disp_2nd,identity):
    disp_composition = disp_2nd + F.grid_sample(disp_1st,disp_2nd.permute(0,2,3,4,1)+identity)
    return disp_composition


def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))


def nnUNetNorm(img):
    mask = img>0
    mean_intensity = img[mask].mean()
    std_intensity = img[mask].std()
    img = (img - mean_intensity) / (std_intensity + 1e-8)
    img[mask == 0] = 0
    return img


def nnUNetNormProps(img,props):
    mean_intensity = props['mean']
    std_intensity = props['sd']
    lower_bound = props['percentile_00_5']
    upper_bound = props['percentile_99_5']
    img1 = torch.clamp(img, lower_bound, upper_bound)
    img1 = (img1 - mean_intensity) / std_intensity
    
    return img1


def nnUNetCTnorm(img):
    img = torch.clamp(img, -1000, 1500)
    mean_intensity = img.mean()
    std_intensity = img.std()
    lower_bound = torch.quantile(img,0.005)
    upper_bound = torch.quantile(img,0.995)
    img = torch.clamp(img, lower_bound, upper_bound)
    img = (img - mean_intensity) / std_intensity
    return img


def find_rigid_3d(x, y):
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3]-x_mean).t(), (y[:, :3]-y_mean)))
    m = torch.eye(v.shape[0], v.shape[0]).to(x.device)
    m[-1,-1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4).to(x.device)
    T[:3,:3] = rotation
    T[:3, 3] = translation
    return T


def least_trimmed_rigid(fixed_pts, moving_pts, iter=5):
    idx = torch.arange(fixed_pts.shape[0]).to(fixed_pts.device)
    for i in range(iter):
        x = find_rigid_3d(fixed_pts[idx,:], moving_pts[idx,:]).t()
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x), 2), 1))
        _, idx = torch.topk(residual, fixed_pts.shape[0]//2, largest=False)
    return x.t()


def compute_steps_for_sliding_window(patch_size, image_size, step_size=.5):

#-> List[List[int]]:
#        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
#        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps



def get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return torch.from_numpy(gaussian_importance_map).unsqueeze(0).unsqueeze(0).half().cuda()


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask
    

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


import numpy as np


def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

class GaussianSmoothing(nn.Module):
    def __init__(self, sigma):
        super(GaussianSmoothing, self).__init__()
        
        sigma = torch.tensor([sigma])
        N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
        weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N), 2) / (2 * torch.pow(sigma, 2)))
        weight /= weight.sum()
        
        self.weight = weight
        
    def forward(self, x):
        device = x.device
        
        x = filter1D(x, self.weight.to(device), 0)
        x = filter1D(x, self.weight.to(device), 1)
        x = filter1D(x, self.weight.to(device), 2)
        
        return x
    
def kovesi_spline(sigma,n=4):
    w_ideal = np.sqrt(12*sigma**2/n+1)
    w_u = int(np.ceil((w_ideal-1)/2)*2+1)
    w_l = max(w_u-2,1)
    m = int(np.round((12*sigma**2-n*w_l**2-4*n*w_l-3*n)/(-4*w_l-4)))
    #print(w_ideal,w_l,m)
    avg = []
    for i in range(m):
        if(w_l>1):
            avg.append(nn.AvgPool3d(w_l,stride=1,padding=(w_l-1)//2))
    for i in range(n-m):
        avg.append(nn.AvgPool3d(w_u,stride=1,padding=(w_u-1)//2))
    avg1 = nn.Sequential(*avg)
    return avg1

def find_rigid_3d(x, y):
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3]-x_mean).t(), (y[:, :3]-y_mean)))
    m = torch.eye(v.shape[0], v.shape[0]).to(x.device)
    m[-1,-1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4).to(x.device)
    T[:3,:3] = rotation
    T[:3, 3] = translation
    return T

def least_trimmed_rigid(fixed_pts, moving_pts, iter=5):
    idx = torch.arange(fixed_pts.shape[0]).to(fixed_pts.device)
    for i in range(iter):
        x = find_rigid_3d(fixed_pts[idx,:], moving_pts[idx,:]).t()
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x), 2), 1))
        _, idx = torch.topk(residual, fixed_pts.shape[0]//2, largest=False)
    return x.t()
#sigmas = torch.tensor([.4,.55,.7,.85,1,1.15,1.4,1.6])
#sigmas2 = torch.tensor([1.2,1.4,1.6,1.9,2.3,2.7,3.2,3.8])
#import matplotlib.pyplot as plt

#for sigma in sigmas2:
#    print(sigma)
#    avg1 = kovesi_spline(sigma,4)
#    print(avg1)
#    empty = torch.zeros(1,1,25,25,25)
#    empty[0,0,12,12,12] = 1
#    empty_f = avg1(empty)
#    gauss = GaussianSmoothing(sigma)
#    print(gauss.weight.shape)
#    empty_g = gauss(empty)
#    plt.imshow(torch.cat((empty_f[0,0,12],empty_g[0,0,12]),1))
#    plt.colorbar()
#    plt.show()
    
