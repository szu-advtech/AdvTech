import jittor as jt
from jittor import Module
class IDRLoss(Module):
    def __init__(self, eikonal_weight, mask_weight, alpha):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.l1_loss = jt.nn.L1Loss()

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return 0.0
        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return 0.0

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = (1-(network_object_mask & object_mask)).bool()
        if mask.sum() == 0:
            return 0.0
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        # print("sdf_pred.size(): ", sdf_pred.shape)
        # print("mask.sum() = ", mask.sum())
        mask_loss = (1 / self.alpha) * jt.nn.binary_cross_entropy_with_logits(sdf_pred.squeeze(1), gt)
        return mask_loss

    def execute(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
        }
