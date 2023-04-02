import torch as nn 





""" Abstract Class method for Variational Autoencoder"""
class BaseVAE(nn.Module):
	def __init__(self):







class GeometricScatteringVAE(nn.Module):
	def __init__(self, 
		adjacency_matrix:nn.Tensor,
		n_layers:int = 2,
		n_scales:int = 4,
		operator_type:str = 'geometric',
		) -> None:





	def forward(self, x:nn.Tensor)->nn.Tensor:
		
