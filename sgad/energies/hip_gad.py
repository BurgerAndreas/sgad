import torch
from typing import Optional

from hip.equiformer_torch_calculator import EquiformerTorchCalculator


class HIPGADEnergy(torch.nn.Module):
    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        tau: float = 1e-3,
        alpha: float = 1e3,
        device: str = "cpu",
    ):
        super().__init__()
        # if model_ckpt is None:
        #     model_ckpt = hf_hub_download(
        #         repo_id="facebook/sgad", filename="esen_spice.pt"
        #     )
        self.calculator = EquiformerTorchCalculator(
            checkpoint_path=model_ckpt,
            hessian_method="predict",
            device=device,
        )
        
        self.tau = tau  # temperature
        self.alpha = alpha  # regularization strength
        self.device = device
        
        # Set atomic numbers for compatibility with existing code
        self.atomic_numbers = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35, 53], dtype=torch.long)
    
    def convert_batch_to_hip_format(self, batch):
        """Convert the batch format to HIP-compatible format"""
        # Extract positions and atomic numbers
        positions = batch["positions"]
        atomic_numbers = batch["node_attrs"].argmax(dim=-1)
        
        # Get batch pointers for splitting molecules
        batch_ptr = batch["ptr"]
        
        # Convert to HIP format for each molecule in the batch
        hip_batch_list = []
        for i in range(len(batch_ptr) - 1):
            start_idx = batch_ptr[i]
            end_idx = batch_ptr[i + 1]
            
            mol_positions = positions[start_idx:end_idx]
            mol_atomic_numbers = atomic_numbers[start_idx:end_idx]
            
            # Create HIP-compatible data structure
            hip_data = {
                "pos": mol_positions,
                "z": mol_atomic_numbers,
                "natoms": torch.tensor([len(mol_atomic_numbers)], dtype=torch.long),
                "cell": None,
                "pbc": torch.tensor(False, dtype=torch.bool),
            }
            hip_batch_list.append(hip_data)
        
        return hip_batch_list
    
    def __call__(self, batch, regularize: Optional[bool] = None):

        # Convert batch to HIP format
        hip_batch_list = self.convert_batch_to_hip_format(batch)
        
        output_dict = {}
        all_energies = []
        all_forces = []
        all_gad = []
        
        # Process each molecule in the batch
        all_hessians = []
        for hip_data in hip_batch_list:
            # Get predictions from HIP calculator
            results = self.calculator.predict(
                coords=hip_data["pos"],
                atomic_nums=hip_data["z"]
            )
            
            all_energies.append(results["energy"])
            all_forces.append(results["forces"])
            all_hessians.append(results["hessian"])
            
            # Get GAD (Gentlest Ascent Dynamics) from HIP
            gad_results = self.calculator.get_gad(hip_data)
            all_gad.append(gad_results["gad"])
        
        # Concatenate results for the batch
        output_dict["energy_physical"] = torch.cat(all_energies, dim=0)
        output_dict["forces_physical"] = torch.cat(all_forces, dim=0)
        output_dict["gad"] = torch.cat(all_gad, dim=0)
        output_dict["hessian"] = torch.stack(all_hessians, dim=0)  # Stack hessians for batch
        
        # Apply temperature scaling to forces
        output_dict["forces"] = output_dict["gad"] / self.tau
        
        return output_dict