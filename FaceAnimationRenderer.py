import os
import math
import cv2
import torch
import numpy as np
import torch.nn as nn
import pytorch3d.transforms
import pytorch3d.utils
import torch
from torchvision.transforms.functional import gaussian_blur
import pytorch3d
from pytorch3d.renderer import cameras
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)


class FaceAnimationRenderer:
    def __init__(self, model_folder, device, mouth_code_dim, eyes_code_dim, pose_code_dim, w=512, h=512, fov=45,
                 projection=None, mouth_gen=None, eye_gen=None, pose_gen=None, mouth_roi=(312, 560, 400, 380), eye_roi=(312, 260, 400, 300), neural_renderer=None,
                 post_rotation=torch.eye(3, 3), post_translation=torch.zeros(1, 3), crop_size=None, n90rots_pre=0, n90rots_post=0, background=None):

        self.base_tex = torch.from_numpy(cv2.cvtColor(cv2.imread("{}/mesh/texture.png".format(model_folder)), cv2.COLOR_BGR2RGB)).to(device).to(torch.float) / 255.0
        self.neural_renderer = neural_renderer

        if mouth_gen is not None:
            self.mouth_mdl = mouth_gen.to(device=device)
        else:
            self.mouth_mdl = torch.jit.load("{}/mesh/model1.pth".format(model_folder)).to(device=device)

        if eye_gen is not None:
            self.eye_mdl = eye_gen.to(device=device)
        else:
            self.eye_mdl = torch.jit.load("{}/mesh/model2.pth".format(model_folder)).to(device=device)

        self.mouth_roi = mouth_roi
        self.eye_roi = eye_roi
        self.mouth_code_dim = mouth_code_dim
        self.eyes_code_dim = eyes_code_dim
        self.pose_code_dim = pose_code_dim
        self.pose_gen = pose_gen
        self.post_rotation = post_rotation
        self.post_translation = post_translation
        self.crop_size = crop_size
        self.cropped_roi = None
        self.n90rots_pre = n90rots_pre
        self.n90rots_post = n90rots_post
        self.background = background

        self.w = w
        self.h = h
        self.fov = fov
        self.projection = projection

        shape_files = []
        for file in os.listdir("{}/mesh".format(model_folder)):
            if file.endswith(".obj"):
                shape_files += [file]
        shape_files.sort()

        mean_verts, faces_idx, aux = load_obj("{}/mesh/{}".format(model_folder, shape_files[0]))

        '''
        rotation_center_interpolation_weight = 0.2
        center_of_rotation = torch.mean(mean_verts, dim=0, keepdim=True)
        mean_y = center_of_rotation[0, 1]
        min_y, _ = torch.min(mean_verts[:, 1], dim=0, keepdim=True)
        center_of_rotation[0, 1] = center_of_rotation[0, 1] * (1.0-rotation_center_interpolation_weight) + min_y[0] * rotation_center_interpolation_weight
        mean_verts -= center_of_rotation
        self.post_rotation_translation = torch.zeros_like(center_of_rotation).to(device=device)
        self.post_rotation_translation[0, 1] = (min_y - mean_y)
        '''

        self.post_rotation_translation = -torch.mean(mean_verts, dim=0, keepdim=True).to(device=device) #torch.zeros(1, 3).to(device=device)

        self.verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        self.faces_uvs = faces_idx.verts_idx.to(device) # (F, 3)
        self.faces = faces_idx.verts_idx.to(device)

        shape_verts = []
        for i in range(1, len(shape_files)):
            verts, _, _ = load_obj("{}/mesh/{}".format(model_folder, shape_files[i]))
            shape_verts += [verts.unsqueeze(0)]
        self.shape_verts = torch.cat(shape_verts, dim=0).to(device=device)
        self.mean_verts = mean_verts.to(device=device)

        self.mouth_blending_mask = torch.ones(1, 1, mouth_roi[3]//2, mouth_roi[2]//2).to(device=device)
        self.eye_blending_mask = torch.ones(1, 1, eye_roi[3]//2, eye_roi[2]//2).to(device=device)

        box_filt = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False, padding_mode='zeros').to(device=device)
        box_filt.weight = nn.Parameter(torch.zeros_like(box_filt.weight), requires_grad=False)
        box_filt.weight[0, 0, :, :] = 1.0 / 9.0

        self.box_filt3 = torch.nn.Conv2d(3, 3, 3, 1, bias=False).to(device=device)
        self.box_filt3.weight = nn.Parameter(torch.zeros_like(self.box_filt3.weight), requires_grad=False)
        for i in range(3):
            self.box_filt3.weight[i, i, :, :] = 1.0/9.0

        with torch.no_grad():
            for i in range(100):
                self.mouth_blending_mask = box_filt(self.mouth_blending_mask)
                self.eye_blending_mask = box_filt(self.eye_blending_mask)

            self.mouth_blending_mask = torch.nn.AdaptiveAvgPool2d((mouth_roi[3]-2, mouth_roi[2]-2))(self.mouth_blending_mask)
            self.eye_blending_mask = torch.nn.AdaptiveAvgPool2d((eye_roi[3]-2, eye_roi[2]-2))(self.eye_blending_mask)

            for i in range(100):
                self.mouth_blending_mask = box_filt(self.mouth_blending_mask)
                self.eye_blending_mask = box_filt(self.eye_blending_mask)

            self.mouth_blending_mask = torch.nn.ZeroPad2d(1)(self.mouth_blending_mask).permute(0, 2, 3, 1)
            self.eye_blending_mask = torch.nn.ZeroPad2d(1)(self.eye_blending_mask).permute(0, 2, 3, 1)

        self.curr_verts = self.mean_verts
        self.curr_tex = torch.clone(self.base_tex.unsqueeze(0))
        self.device = device

    def update_expression(self, params, rot_type='euler'):
        with torch.no_grad():
            if rot_type == 'euler':
                mouth_params, eye_params, pose_params = torch.split_with_sizes(params, (self.mouth_code_dim, self.eyes_code_dim, self.pose_code_dim), dim=1)
                if self.pose_gen is not None:
                    pose = self.pose_gen(pose_params)
                else:
                    pose = pose_params
                tRx = pose[0, 0]# * 0
                tRy = pose[0, 1]# * 0
                tRz = pose[0, 2]# * 0
                rotmat = torch.zeros(3, 3)
                sinRx = math.sin(tRx)
                cosRx = math.cos(tRx)
                sinRy = math.sin(tRy)
                cosRy = math.cos(tRy)
                sinRz = math.sin(tRz)
                cosRz = math.cos(tRz)
                rotmat[0, 0] = (cosRy*cosRz+sinRy*sinRx*sinRz)
                rotmat[1, 0] = (cosRx*sinRz)
                rotmat[2, 0] = (-sinRy*cosRz + cosRy*sinRx*sinRz)
                rotmat[0, 1] = (-cosRy*sinRz+sinRy*sinRx*cosRz)
                rotmat[1, 1] = (cosRx*cosRz)
                rotmat[2, 1] = (sinRy*sinRz + cosRy*sinRx*cosRz)
                rotmat[0, 2] = (sinRy*cosRx)
                rotmat[1, 2] = (-sinRx)
                rotmat[2, 2] = (cosRy*cosRx)
            elif rot_type == '6D':
                mouth_params, eye_params, pose_params = torch.split_with_sizes(params, (self.mouth_code_dim, self.eyes_code_dim, 6), dim=1)
                if self.pose_gen is not None:
                    pose = self.pose_gen(pose_params)
                else:
                    pose = pose_params
                    rotmat = pytorch3d.transforms.rotation_6d_to_matrix(pose).squeeze(0)
            else:
                print("Unknown rotation data")
                exit(0)

            tTrans = pose[0, 3:6].unsqueeze(0)# * 0

            self.curr_tex = torch.clone(self.base_tex.unsqueeze(0))

            # update mouth texture
            mouth_tex_data_size = self.mouth_roi[2]*self.mouth_roi[3]*3
            mouth_result = self.mouth_mdl(mouth_params).detach()
            mouth_tex_data = mouth_result[:, :mouth_tex_data_size].view(-1, 3, self.mouth_roi[3], self.mouth_roi[2])
            mouth_tex = mouth_tex_data.permute(0, 2, 3, 1) * 0.5 + 0.5
            x0, x1 = self.mouth_roi[0], self.mouth_roi[0] + self.mouth_roi[2]
            y0, y1 = self.mouth_roi[1], self.mouth_roi[1] + self.mouth_roi[3]
            correction = torch.median(self.curr_tex[:, y0:y1, x0:x1, :] - mouth_tex)
            self.curr_tex[:, y0:y1, x0:x1, :] = (mouth_tex+correction).clamp(-1, 1) * self.mouth_blending_mask + self.curr_tex[:, y0:y1, x0:x1, :] * (1.0 - self.mouth_blending_mask)
            #self.curr_tex[:, y0:y1, x0:x1, 1:2] = self.mouth_blending_mask

            # update eye texture
            eye_tex_data_size = self.eye_roi[2]*self.eye_roi[3]*3
            eye_result = self.eye_mdl(eye_params).detach()
            eye_tex_data = eye_result[:, :eye_tex_data_size].view(-1, 3, self.eye_roi[3], self.eye_roi[2])
            eye_tex = eye_tex_data.permute(0, 2, 3, 1) * 0.5 + 0.5
            x0, x1 = self.eye_roi[0], self.eye_roi[0] + self.eye_roi[2]
            y0, y1 = self.eye_roi[1], self.eye_roi[1] + self.eye_roi[3]
            correction = torch.median(self.curr_tex[:, y0:y1, x0:x1, :] - eye_tex)
            self.curr_tex[:, y0:y1, x0:x1, :] = (eye_tex+correction).clamp(-1, 1) * self.eye_blending_mask + self.curr_tex[:, y0:y1, x0:x1, :] * (1.0 - self.eye_blending_mask)
            #self.curr_tex[:, y0:y1, x0:x1, 0:1] = self.eye_blending_mask

            # update face geometry
            mouth_geo_data = mouth_result[:, mouth_tex_data_size:]
            num_pca_weights = mouth_geo_data.size(1)
            self.curr_verts = self.mean_verts + torch.sum(self.shape_verts[:num_pca_weights, :, :] * mouth_geo_data.squeeze(0).unsqueeze(1).unsqueeze(2), dim=0)

            # rotate & translate vertices
            transform = pytorch3d.transforms.Rotate(rotmat.transpose(0, 1).detach().to(device=self.device))
            self.curr_verts = transform.transform_points(self.curr_verts)
            self.curr_verts += self.post_rotation_translation + tTrans

            # apply extrinsic calibration params
            #transform = pytorch3d.transforms.Rotate(self.post_rotation.transpose(0, 1).detach().to(device=self.device))
            transform = pytorch3d.transforms.Rotate(self.post_rotation.detach().to(device=self.device))
            self.curr_verts = transform.transform_points(self.curr_verts)
            self.curr_verts += self.post_translation.detach().to(device=self.device)

    def render(self):
        return self.render_fov(self.fov, self.w, self.h)

    def render_fov(self, fov=20, w=512, h=512):

        bg_color = (1.0, 1.0, 1.0)
        if self.neural_renderer is not None:
            bg_color = (0.5, 0.5, 0.5)

        with torch.no_grad():
            tex = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=self.curr_tex)
            mesh = Meshes(verts=[self.curr_verts], faces=[self.faces], textures=tex)
            cameras = FoVPerspectiveCameras(znear=1, fov=fov, R=pytorch3d.transforms.euler_angles_to_matrix(torch.FloatTensor([[0, math.radians(180), 0]]), 'XYZ'), device=self.device)
            pytorch3d.renderer.PerspectiveCameras()
            raster_settings = RasterizationSettings(image_size=(h, w), blur_radius=0.0, faces_per_pixel=1)
            lights = AmbientLights(ambient_color=[[1, 1, 1]], device=self.device)
            renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                    shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights, blend_params=pytorch3d.renderer.BlendParams(background_color=bg_color))).eval()
            try:
                images = renderer(mesh).permute(0, 3, 1, 2)
            except:
                images = torch.zeros(1, 4, h, w).to(device=self.device)

            images = images[:, :3, :, :]

            if self.neural_renderer is not None:

                fg_masks = (images != torch.FloatTensor(bg_color).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)).to(dtype=torch.float32)
                fg_mask = torch.logical_and(torch.logical_and(fg_masks[:, 0, :, :], fg_masks[:, 1, :, :]), fg_masks[:, 2, :, :])

                train_batch_mask_eroded3 = torch.zeros_like(fg_mask).to(dtype=torch.float32)
                train_batch_mask_eroded3[fg_mask] = 1.0

                train_batch_mask_dilated3 = torch.zeros_like(fg_mask).to(dtype=torch.float32)
                train_batch_mask_dilated3[fg_mask] = 1.0

                train_batch_mask_eroded3 = train_batch_mask_eroded3.unsqueeze(0).repeat(1, 3, 1, 1)
                train_batch_mask_dilated3 = train_batch_mask_dilated3.unsqueeze(0).repeat(1, 3, 1, 1)

                train_batch_render = images
                train_batch_render_dilated = images

                for i in range(5):
                    train_batch_mask_eroded3 = self.box_filt3(torch.nn.functional.pad(train_batch_mask_eroded3, (1, 1, 1, 1), mode='replicate'))
                    valid_mask = (train_batch_mask_eroded3 >= 1.0)
                    invalid_mask = (train_batch_mask_eroded3 < 1.0)
                    train_batch_mask_eroded3[valid_mask] = 1.0
                    train_batch_mask_eroded3[invalid_mask] = 0.0

                    train_batch_mask_dilated3 = self.box_filt3(torch.nn.functional.pad(train_batch_mask_dilated3, (1, 1, 1, 1), mode='replicate'))
                    valid_mask = (train_batch_mask_dilated3 > 0.0)
                    invalid_mask = (train_batch_mask_dilated3 <= 0.0)
                    train_batch_mask_dilated3[valid_mask] = 1.0
                    train_batch_mask_dilated3[invalid_mask] = 0.0

                for i in range(25):
                    # train_batch_render_dilated = self.poisson_filt3(F.pad(train_batch_render_dilated, (1, 1, 1, 1), mode='replicate'))
                    train_batch_render_dilated = gaussian_blur(train_batch_render_dilated, [3, 3], 1.0)
                    train_batch_render_dilated[train_batch_mask_eroded3 > 0] = train_batch_render[train_batch_mask_eroded3 > 0]
                    if i < (25 - 2):
                        train_batch_render_dilated[train_batch_mask_dilated3 <= 0] = train_batch_render[train_batch_mask_dilated3 <= 0]

                images = train_batch_render_dilated

                render_in = (images[:, :3, :, :] - 0.5) * 2

                if self.n90rots_pre != 0:
                    render_in = torch.rot90(render_in, k=self.n90rots_pre, dims=(3, 2))

                #render_in = torch.rot90((images[:, :3, :, :] - 0.5) * 2, k=1, dims=(2, 3))

                render_bg = torch.ones_like(render_in)
                render_out = self.neural_renderer(render_in)
                refine_rgb = torch.tanh(render_out[:, :3, :, :])
                blend_w = torch.softmax(render_out[:, 3:, :, :], dim=1)

                result = (render_in * blend_w[:, 0:1, :, :] + refine_rgb * blend_w[:, 1:2, :, :] + render_bg * blend_w[:, 2:3, :, :]) * 0.5 + 0.5

                if self.n90rots_pre != 0:
                    result = torch.rot90(result, k=-self.n90rots_pre, dims=(3, 2))

                images[:, :3, :, :] = result

        return images