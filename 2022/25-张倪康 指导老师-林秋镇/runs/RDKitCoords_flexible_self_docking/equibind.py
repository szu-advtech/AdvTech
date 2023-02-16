class EquiBind(nn.Module):

    def __init__(self, device='cuda:0', debug=False, use_evolved_lig=False, evolve_only=False, **kwargs):
        super(EquiBind, self).__init__()
        self.debug = debug
        self.evolve_only = evolve_only
        self.use_evolved_lig = use_evolved_lig
        self.device = device
        self.iegmn = IEGMN(device=self.device, debug=self.debug, evolve_only=self.evolve_only, **kwargs)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, lig_graph, rec_graph, geometry_graph=None, complex_names=None, epoch=0):
        if self.debug: log(complex_names)
        predicted_ligs_coords_list = []
        outputs = self.iegmn(lig_graph, rec_graph, geometry_graph, complex_names, epoch)
        evolved_ligs = outputs[4]
        if self.evolve_only:
            return evolved_ligs, outputs[2], outputs[3], outputs[0], outputs[1], outputs[5]
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        for idx in range(len(ligs_node_idx) - 1):
            start = ligs_node_idx[idx]
            end = ligs_node_idx[idx + 1]
            orig_coords_lig = lig_graph.ndata['new_x'][start:end]
            rotation = outputs[0][idx]
            translation = outputs[1][idx]
            assert translation.shape[0] == 1 and translation.shape[1] == 3

            if self.use_evolved_lig:
                predicted_coords = (rotation @ evolved_ligs[idx].t()).t() + translation  # (n,3)
            else:
                predicted_coords = (rotation @ orig_coords_lig.t()).t() + translation  # (n,3)
            if self.debug:
                log('rotation', rotation)
                log('rotation @ rotation.t() - eye(3)', rotation @ rotation.t() - torch.eye(3).to(self.device))
                log('translation', translation)
                log('\n ---> predicted_coords mean - true ligand mean ',
                    predicted_coords.mean(dim=0) - lig_graph.ndata['x'][
                                                   start:end].mean(dim=0), '\n')
            predicted_ligs_coords_list.append(predicted_coords)
        #torch.save({'predictions': predicted_ligs_coords_list, 'names': complex_names})
        return predicted_ligs_coords_list, outputs[2], outputs[3], outputs[0], outputs[1], outputs[5]

    def __repr__(self):
        return "EquiBind " + str(self.__dict__)
