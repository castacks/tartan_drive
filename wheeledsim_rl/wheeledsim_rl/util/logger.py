import torch
import pandas as pd
from tabulate import tabulate

class Logger:
	"""
	Use to store stats of various RL experiments as a dict and stuff.
	"""
	def __init__(self):
		self.data = {}

	def record_item(self, field, data, prefix=None):
		if prefix:
			key = "{}/{}".format(prefix, field)
		else:
			key = "/{}".format(field)
		if type(data) == torch.Tensor:
			data = data.cpu().detach().numpy()
		self.data[key] = data

	def record_tensor(self, field, t, prefix=None):
		"""
		Reports mean, median, min, max and std of a 1D tensor.
		"""
		stats = ['Mean', 'Median', 'Min', 'Max', 'Std']
		if prefix:
			keys = ["{}/{} {}".format(prefix, field, s) for s in (stats)]
		else:	
			keys = ["/{} {}".format(field, s) for s in (stats)]

		#Note: If we recieve long tensor, we need to cast to float.
		tensor=t.float().clone().detach().cpu()
		values = [tensor.mean(), tensor.median(), tensor.min(), tensor.max(), tensor.std()]
		for k, v in zip(keys, values):
			self.data[k] = v.numpy()

	def get(self, prefix = "", field="", default=0):
		key = '{}/{}'.format(prefix, field)
		return self.data.get(key, 0)

	def dump_dataframe(self):
		return pd.DataFrame({k:[v] for k, v in self.data.items()})

	def print_data(self, fmt='psql'):
		"""
		Ok, we want to print keys in the following order:
		First, print out the fields with no headers. Then print out every field with each header. Within a header, print alphabetically. I did this by imposing a '/' char at the front of every prefixed field.
		"""
		sk = sorted(self.data.keys())
		print(tabulate([[k, self.data[k]] for k in sk], tablefmt=fmt))
