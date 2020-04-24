import torch



def comp(a,b,A,B):
    batch = a.size(0)
    a_ = a.unsqueeze(1).contiguous().view(batch,1,-1)
    b_ = b.unsqueeze(1).contiguous().view(batch,1,-1)
    c_ = torch.cat((a_,b_),1)
    m = c_.max(1)[0].unsqueeze(1).expand_as(c_)
    m = (c_==m).float()
    m1 = m.permute(0,2,1)
    k = m1[...,0]
    j = m1[...,1]
    z = ((k*j)!=1).float()
    j = z*j
    m1 = torch.cat((k,j),1).unsqueeze(1).view_as(m)

    A_ = A.unsqueeze(1).contiguous().view(batch,1,-1)
    B_ = B.unsqueeze(1).contiguous().view(batch,1,-1)
    C_ = torch.cat((A_,B_),1).permute(0,2,1)
    m1 = m1.long().permute(0,2,1)
    res = C_[m1.long()==1].view_as(a)

    return res


class LeftPool(torch.autograd.Function):
    
    def forward(self, input_):
        self.save_for_backward(input_.clone())
        output = torch.zeros_like(input_)
        batch = input_.size(0)
        width = input_.size(3)
  
        input_tmp = input_.select(3, width-1)
        output.select(3,width-1).copy_(input_tmp)
        
        for idx in range(1, width):
            input_tmp = input_.select(3,width-idx-1)
            output_tmp = output.select(3,width-idx)
            cmp_tmp = torch.cat((input_tmp.view(batch,1,-1),output_tmp.view(batch,1,-1)),1).max(1)[0]
            output.select(3,width-idx-1).copy_(cmp_tmp.view_as(input_tmp))
         
        return output
    
    def backward(self,grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        w = input_.size(3)
        batch = input_.size(0)
        
        output_tmp = res.select(3, w-1)
        grad_output_tmp = grad_output.select(3, w-1)
        output_tmp.copy_(grad_output_tmp)
        
        input_tmp = input_.select(3, w-1)
        output.select(3,w-1).copy_(input_tmp)
        
        for idx in range(1, w):
            
            input_tmp = input_.select(3, w-idx-1)
            output_tmp = output.select(3,w-idx)
            cmp_tmp = torch.cat((input_tmp.view(batch,1,-1),output_tmp.view(batch,1,-1)),1).max(1)[0]
            output.select(3,w-idx-1).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(3, w-idx-1)
            res_tmp = res.select(3,w-idx)
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(3,w-idx-1).copy_(com_tmp)
        return res
    
class RightPool(torch.autograd.Function):
    def forward(self, input_):
        self.save_for_backward(input_)
        
        output = torch.zeros_like(input_)
        width = input_.size(3)
        batch = input_.size(0)
 
        input_tmp = input_.select(3, 0)
        output.select(3,0).copy_(input_tmp)
        
        for idx in range(1, width):
            input_tmp = input_.select(3,idx)
            output_tmp = output.select(3,idx-1)

            cmp_tmp = torch.cat((input_tmp.view(batch,1,-1),output_tmp.view(batch,1,-1)),1).max(1)[0]
            output.select(3,idx).copy_(cmp_tmp.view_as(input_tmp))
        return output
    
    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        w = input_.size(3)
        batch = input_.size(0)
        
        output_tmp = res.select(3, 0)
        grad_output_tmp = grad_output.select(3, 0)
        output_tmp.copy_(grad_output_tmp)
        
        input_tmp = input_.select(3, 0)
        output.select(3,0).copy_(input_tmp)
        
        for idx in range(1, w):
            input_tmp = input_.select(3,idx)
            output_tmp = output.select(3,idx-1)
            cmp_tmp = torch.cat((input_tmp.view(batch,1,-1),output_tmp.view(batch,1,-1)),1).max(1)[0]
            output.select(3,idx).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(3, idx)
            res_tmp = res.select(3,idx-1)
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(3,idx).copy_(com_tmp)
        return res
            
class TopPool(torch.autograd.Function):
    
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.zeros_like(input_)
        height = output.size(2)
        batch = input_.size(0)

        input_tmp = input_.select(2, height-1)
        output.select(2,height-1).copy_(input_tmp)
        
        for idx in range(1, height):
            input_tmp = input_.select(2, height-idx-1)
            output_tmp = output.select(2,height-idx)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]

            output.select(2, height-idx-1).copy_(cmp_tmp.view_as(input_tmp))
        return output
    
    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        height = output.size(2)
        batch = input_.size(0)
        #copy the last row
        input_tmp = input_.select(2, height-1)
        output.select(2,height-1).copy_(input_tmp)
        
        grad_tmp = grad_output.select(2, height-1)
        res.select(2,height-1).copy_(grad_tmp)
        for idx in range(1, height):
            input_tmp = input_.select(2, height-idx-1)
            output_tmp = output.select(2,height-idx)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(2, height-idx-1).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(2, height-idx-1)
            res_tmp = res.select(2,height-idx)
            
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(2,height-idx-1).copy_(com_tmp)
        return res
    
class BottomPool(torch.autograd.Function):
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.zeros_like(input_)
        height = output.size(2)
        batch = output.size(0)

        input_tmp = input_.select(2, 0)
        output.select(2,0).copy_(input_tmp)
        
        for idx in range(1, height):
            input_tmp = input_.select(2, idx)
            output_tmp = output.select(2,idx-1)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))
        return output
    
    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        height = output.size(2)
        batch = output.size(0)
 
        input_tmp = input_.select(2,0)
        output.select(2,0).copy_(input_tmp)
        
        grad_tmp = grad_output.select(2,0)
        res.select(2,0).copy_(grad_tmp)
        
        for idx in range(1, height):
            input_tmp = input_.select(2, idx)
            output_tmp = output.select(2,idx-1)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(2, idx)
            res_tmp = res.select(2,idx-1)
            
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(2,idx).copy_(com_tmp)
        return res