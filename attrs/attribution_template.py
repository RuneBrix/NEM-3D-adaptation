import torch
import numpy as np
class attribution_template_class:
    def __init__(self, model, train_data = None, use_predicted_labels = False, store_path = None):
        self.attr_name = "template"
        self.model = model
        self.use_predicted_labels = use_predicted_labels
        self.method = None
    def gen_attr(self,  X, y):  
        pass
         
    def gen_attrs(self,  data):
        self.model.eval().cuda()
        attrs = []
        for i ,(X, y,) in enumerate(data): 
                name =  f"{i}"
                X = X.cuda()
                
                if self.use_predicted_labels:
                    y = torch.argmax(self.model(X))
                attr = self.gen_attr(X, y)
                attrs.append((
                    attr,
                    y.cpu().numpy(),
                    name))
        return attrs
    def rank_attr(self,attr):
        flat_array = attr.reshape(-1)
    
        ranks = np.zeros_like(flat_array, dtype=int)
        sorted_indices = np.argsort(flat_array)
        ranks[sorted_indices] = np.arange(flat_array.shape[0])
    
        return ranks.reshape(attr.shape)