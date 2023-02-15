import jittor as jt
from utils import rend_util
from jittor import Module
class RayTracing(Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_secant_steps=8,
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

    def execute(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions
                ):

        batch_size, num_pixels, _ = ray_directions.shape

        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)
        
        print("cam_loc")
        print(cam_loc)
        
        print("ray_directions")
        print(ray_directions.shape)
        
        print(ray_directions)
        print("mask_intersect")
        print(jt.sum(mask_intersect))
        print(mask_intersect)
        print("sphere_intersections")
        print(sphere_intersections.shape)
        
        print(sphere_intersections)
        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)

        network_object_mask = (acc_start_dis < acc_end_dis)
        print("acc_start_dis")
        print(acc_start_dis)
        print("acc_end_dis")
        print(acc_end_dis)
        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = jt.zeros_like(sampler_mask).bool()
        if sampler_mask.sum() > 0:
            print("66666666666666666sampler_mask.sum() = ", sampler_mask.sum())
            sampler_min_max = jt.zeros((batch_size, num_pixels, 2))
            sampler_min_max = sampler_min_max.reshape(-1, 2)
            sampler_min_max[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max[sampler_mask, 1] = acc_end_dis[sampler_mask]
            # print("sum 999999999999sampler_min_max")
            
            # print(sampler_min_max.reshape(-1, 2)[sampler_mask])
            sampler_min_max = sampler_min_max.reshape((batch_size, num_pixels, 2))
            # print("acc_start_dis[sampler_mask]")
            # print(acc_start_dis[sampler_mask])
            
            # print("acc_end_dis[sampler_mask]")
            # print(acc_end_dis[sampler_mask])
            # print("sampler_min_max.shape", sampler_min_max.shape)
            # print(jt.sum(sampler_min_max))

            
            # print(sampler_min_max)
            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )
            # print("tyep(acc_start_dis) : ", type(acc_start_dis))
            # print("tyep(network_object_mask) : ", type(network_object_mask))
            # print("acc_start_dis.shape: ", acc_start_dis.shape)
            # print("network_object_mask.shape: ", network_object_mask.shape)
            
            # print("tyep(sampler_net_obj_mask) : ", type(sampler_net_obj_mask))
            # print("sampler_net_obj_mask.shape: ", sampler_net_obj_mask.shape)
            # print("sampler_net_obj_mask[0]", sampler_net_obj_mask[0])
            xx = sampler_net_obj_mask[0]
            # print("sampler_net_obj_mask.sum() = ", sampler_net_obj_mask.sum())
            
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]
            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            # network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]
        # print("network_object_mask.shape", network_object_mask.shape)
        # print("type(network_object_mask): ", network_object_mask)
        # print("network_object_mask.sum()", jt.sum(network_object_mask))
        # print("len(network_object_mask)", len(network_object_mask))
        # print("sampler_net_obj_mask.sum()", sampler_net_obj_mask.sum())
        # print("sampler_mask.sum()", sampler_mask.sum())
        
        print('----------------------------------------------------------------')
        print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
              .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        print('----------------------------------------------------------------')

        if not self.is_training:
            return curr_start_points, \
                   network_object_mask, \
                   acc_start_dis

        ray_directions = ray_directions.reshape(-1,3)
        mask_intersect = mask_intersect.reshape(-1)

        in_mask = jt.logical_not(network_object_mask) & object_mask & jt.logical_not(sampler_mask)
        out_mask = jt.logical_not(object_mask) & jt.logical_not(sampler_mask)

        mask_left_out = (in_mask | out_mask) & jt.logical_not(mask_intersect)
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            # print("-jt.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).shape", (-jt.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1))).shape)
            acc_start_dis[mask_left_out] = -jt.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze(1).squeeze(1)
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]

            min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        return curr_start_points, \
               network_object_mask, \
               acc_start_dis


    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()
        print("unfinished_mask_start")
        print(unfinished_mask_start.shape)
        print(jt.sum(unfinished_mask_start))
        print(unfinished_mask_start)
        # Initialize start current points
        curr_start_points = jt.zeros((batch_size * num_pixels, 3)).float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = jt.zeros(batch_size * num_pixels).float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0]
        print("acc_start_dis[unfinished_mask_start]")
        print(acc_start_dis[unfinished_mask_start])
        
        # Initialize end current points
        curr_end_points = jt.zeros((batch_size * num_pixels, 3)).float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        acc_end_dis = jt.zeros(batch_size * num_pixels).float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1]

        print("acc_end_dis[unfinished_mask_end]")
        print(acc_end_dis[unfinished_mask_end])
        
        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = jt.zeros_like(acc_start_dis)
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        print("next_sdf_start[unfinished_mask_start]")
        print(next_sdf_start[unfinished_mask_start])
        next_sdf_end = jt.zeros_like(acc_end_dis)
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])
        
        print("next_sdf_end[unfinished_mask_end]")
        print(next_sdf_end[unfinished_mask_end])
        while True:
            # Update sdf
            curr_sdf_start = jt.zeros_like(acc_start_dis)
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = jt.zeros_like(acc_end_dis)
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)
            print("unfinished_mask_start.sum()")
            print(unfinished_mask_start.sum())
            
            print("unfinished_mask_end.sum()")
            print(unfinished_mask_end.sum())
            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end
            # print("acc_start_dis")
            # print(acc_start_dis)
            
            # print("acc_end_dis")
            # print(acc_end_dis)
            
            # print("curr_sdf_start")
            # print(curr_sdf_start)
            
            
            # print("curr_sdf_end")
            # print(curr_sdf_end)
            # Update points
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = jt.zeros_like(acc_start_dis)
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = jt.zeros_like(acc_end_dis)
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = jt.zeros((n_total_pxl, 3)).float()
        sampler_dists = jt.zeros(n_total_pxl).float()

        intervals_dist = jt.linspace(0, 1, steps=self.n_steps).view(1, 1, -1)
        # print("intervals_dist:")
        # print(intervals_dist)
        # print("jt.sum(intervals_dist)", jt.sum(intervals_dist))     
        # print("sampler_min_max")
        # print(sampler_min_max)
        
        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        # print("pts_intervals: ")
        # print(pts_intervals)
        # print("jt.sum(pts_intervals)", jt.sum(pts_intervals))     
        
        points = cam_loc.reshape(batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        mask_intersect_idx = jt.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in jt.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = jt.concat(sdf_val_all).reshape(-1, self.n_steps)
        # print("sdf_val[0]")
        # print(sdf_val[0])
        
        tmp = jt.nn.sign(sdf_val) * jt.arange(self.n_steps, 0, -1).float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = jt.argmin(tmp, -1)[0]
        
        
        # print("sampler_pts_ind")
        # print(sampler_pts_ind)
        # print(jt.argmin(tmp, -1))
        sampler_pts[mask_intersect_idx] = points[jt.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[jt.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[jt.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = jt.logical_not((true_surface_pts & net_surface_pts))
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = jt.argmin(sdf_val[p_out_mask, :], -1)[0]
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][jt.arange(n_p_out.item()), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][jt.arange(n_p_out.item()), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[jt.logical_not(net_surface_pts)]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.is_training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[jt.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[jt.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][jt.arange(n_secant_pts.item()), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][jt.arange(n_secant_pts.item()), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = jt.linspace(0.0, 1.0,n)
        steps = jt.zeros(n).uniform_(0.0, 1.0)
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points.item(), 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in jt.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = jt.concat(mask_sdf_all).reshape(-1, n)

        # min_vals, min_idx = mask_sdf_all.min(-1)
        min_idx, min_vals = jt.arg_reduce(mask_sdf_all, 'min', dim=-1, keepdims=False)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[jt.arange(0, n_mask_points.item()), min_idx]
        min_mask_dist = steps.reshape(-1, n)[jt.arange(0, n_mask_points.item()), min_idx]

        return min_mask_points, min_mask_dist
