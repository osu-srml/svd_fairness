import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from scipy.optimize import fsolve
from tqdm import tqdm

def get_sigma_fair_1(sigma, k, lam):
    result = (sigma * k) / (lam + k)
    return result

def func_1(lam, k, sigma, c):
    return sum([get_sigma_fair_1(sigma[i].item(), k[i].item(), lam.item()) ** 2 for i in range(k.shape[0])]) - c

def get_lambda_1(k, sigma, c):
    lambad_initial_guess = 1.0
    lambda_solution = fsolve(func_1, lambad_initial_guess, args=(k, sigma, c))
    lambda_optimal = lambda_solution[0]
    return lambda_optimal

def loss_1_fn(w, x_1, x_2):
    x_1_bar = torch.mean(x_1, dim=0).unsqueeze(0)
    x_2_bar = torch.mean(x_2, dim=0).unsqueeze(0)
    loss = torch.sum((torch.matmul(x_1_bar, w.T) - torch.matmul(x_2_bar, w.T)) ** 2)
    return loss / w.size()[0]

def loss_2_fn(w, x_1, x_2):
    x_1_bar = torch.mean(x_1, dim=0).unsqueeze(0)
    x_2_bar = torch.mean(x_2, dim=0).unsqueeze(0)
    cov_1 = torch.matmul((x_1 - x_1_bar).T, (x_1 - x_1_bar)) / x_1.size()[0]
    cov_2 = torch.matmul((x_2 - x_2_bar).T, (x_2 - x_2_bar)) / x_2.size()[0]
    loss = torch.sum((torch.matmul(w, torch.matmul(cov_1, w.T)) - torch.matmul(w, torch.matmul(cov_2, w.T))) ** 2)
    return loss / (w.size()[0] * w.size()[0])

def loss_3_fn(w_1, w_2, x):
    loss = torch.sum((torch.matmul(x, w_1.T) - torch.matmul(x, w_2.T)) ** 2)
    return loss / (x.size()[0] * w_1.size()[0])

def gradient_descent(weight, x_1, x_2):
    fair_weight = weight.clone()
    fair_weight.requires_grad = True
    
    x = torch.cat([x_1, x_2], dim=0)
    
    loss_1_list = []
    loss_2_list = []
    loss_3_list = []
    optimizer = torch.optim.SGD([fair_weight], lr=1e-4)
    for i in range(10000):
        optimizer.zero_grad()
        #loss = loss_1_fn(fair_weight, x_1, x_2) + loss_2_fn(fair_weight, x_1, x_2) + loss_3_fn(fair_weight, weight, x)
        loss_1 = loss_1_fn(fair_weight, x_1, x_2)
        loss_2 = loss_2_fn(fair_weight, x_1, x_2)
        loss_3 = loss_3_fn(fair_weight, weight, x)
        loss = loss_1 + loss_2 + loss_3
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss_1 = loss_1.item()
            initial_loss_2 = loss_2.item()
            initial_loss_3 = loss_3.item()
            print("Initial loss_1 = {}, loss_2 = {}, loss_3 = {}".format(initial_loss_1, initial_loss_2, initial_loss_3))
        loss_1_list.append(loss_1.item())
        loss_2_list.append(loss_2.item())
        loss_3_list.append(loss_3.item())
        plt.plot(range(1, i + 2), loss_1_list)
        plt.grid(True)
        plt.savefig("loss_1.png")
        plt.close()
        plt.plot(range(1, i + 2), loss_2_list)
        plt.grid(True)
        plt.savefig("loss_2.png")
        plt.close()
        plt.plot(range(1, i + 2), loss_3_list)
        plt.grid(True)
        plt.savefig("loss_3.png")
        plt.close()
    
    print("loss_1 = {}, loss_2 = {}, loss_3 = {}".format(loss_1, loss_2, loss_3))
    print("ratio loss_1 = {}, loss_2 = {}, loss_3 = {}".format(initial_loss_1 / loss_1, initial_loss_2 / loss_2, 
                                                               initial_loss_3 / loss_3))
    return fair_weight

def expectation_constrain(weight, x_1, x_2, c):
    x_1_bar = torch.mean(x_1, dim=0).unsqueeze(0)
    x_2_bar = torch.mean(x_2, dim=0).unsqueeze(0)
    
    x = torch.cat([x_1, x_2], dim=0)
    
    xxt = torch.matmul((x_1_bar - x_2_bar).T, (x_1_bar - x_2_bar))
    s = torch.linalg.cholesky(xxt + torch.eye(xxt.size()[0]) * 1e-5)
    ws = torch.matmul(weight, s)
    u, sig, vt = torch.linalg.svd(ws, full_matrices=False)
    
    k = torch.zeros(sig.size()[0])
    for i in range(sig.size()[0]):
        v_i = vt[i, :].unsqueeze(0).T
        k[i] = torch.matmul(v_i.T, 
                            torch.matmul(torch.linalg.inv(s), 
                                         torch.matmul(x.T, 
                                                      torch.matmul(x, 
                                                                   torch.matmul(torch.linalg.inv(s).T, 
                                                                                v_i)))))
        #k[i] = torch.matmul(v_i.T,
        #                    torch.matmul(torch.linalg.inv(s),
        #                                 torch.matmul(torch.linalg.inv(s).T, v_i)))
    lam = get_lambda_1(k.clone().cpu().numpy(), sig.clone().cpu().numpy(), torch.sum(sig ** 2).item() / c)
    sig_fair = get_sigma_fair_1(sig, k, lam)
    
    fair_weight_1 = torch.matmul(torch.matmul(u, torch.matmul(torch.diag(sig_fair), vt)), torch.linalg.inv(s))
    return fair_weight_1

def get_sigma_fair_2(sigma, k, lam):
    inter = (9 * lam ** 2 * k * sigma +  \
        np.sqrt(3) * np.sqrt(2 * lam ** 3 * k ** 3 + 27 * lam ** 4 * k * sigma ** 2 + 1e-5)) ** (1 / 3)
    result = - k / ((6 ** (1 / 3)) * inter + 1e-5) + inter / (6 ** (2 / 3) * lam)
    return result

def func_2(lam, k, sigma, c):
    return sum([get_sigma_fair_2(sigma[i].item(), k[i].item(), lam.item()) ** 4 for i in range(k.shape[0])]) - c

def get_lambda_2(k, sigma, c):
    lambad_initial_guess = 1.0
    lambda_solution = fsolve(func_2, lambad_initial_guess, args=(k, sigma, c))
    lambda_optimal = lambda_solution[0]
    return lambda_optimal

def covaraince_constrain(weight, x_1, x_2, c):
    x_1_bar = torch.mean(x_1, dim=0).unsqueeze(0)
    x_1_normalized = x_1 - x_1_bar
    x_2_bar = torch.mean(x_2, dim=0).unsqueeze(0)
    x_2_normalized = x_2 - x_2_bar
    
    x = torch.cat([x_1, x_2], dim=0)
    
    xxt = (1 / (x_1.size()[0] - 1)) * torch.matmul(x_1_normalized.T, x_1_normalized) - \
        (1 / (x_2.size()[0] - 1)) * torch.matmul(x_2_normalized.T, x_2_normalized)
    eig_values, eig_vectors = torch.linalg.eig(xxt)
    s = torch.matmul(eig_vectors.real, torch.diag(torch.sqrt(torch.abs(eig_values.real))))
    ws = torch.matmul(weight, s)
    u, sig, vt = torch.linalg.svd(ws, full_matrices=False)
    
    k = torch.zeros(sig.size()[0])
    for i in range(sig.size()[0]):
        v_i = vt[i, :].unsqueeze(0).T
        k[i] = torch.matmul(v_i.T, 
                            torch.matmul(torch.linalg.pinv(s), 
                                         torch.matmul(x.T, 
                                                      torch.matmul(x, 
                                                                   torch.matmul(torch.linalg.pinv(s).T, 
                                                                                v_i)))))
    lam = get_lambda_2(k.clone().cpu().numpy(), sig.clone().cpu().numpy(), torch.sum(sig ** 4).item() / c)
    sig_fair = torch.zeros(sig.size()[0])
    for i in range(sig.size()[0]):
        sig_fair[i] = get_sigma_fair_2(sig[i].item(), k[i].item(), lam.item())
    
    fair_weight_2 = torch.matmul(torch.matmul(u, torch.matmul(torch.diag(sig_fair), vt)), torch.linalg.pinv(s))
    return fair_weight_2
    

def compress(model, train_X, train_a, train_y, c_1, c_2):    
    group_1_indices = (train_a == 0)
    group_2_indices = (train_a == 1)
    
    group_1_X = train_X[group_1_indices]
    group_1_a = train_a[group_1_indices]
    group_1_y = train_y[group_1_indices]
    group_2_X = train_X[group_2_indices]
    group_2_a = train_a[group_2_indices]
    group_2_y = train_y[group_2_indices]
    
    group_1_dataset = torch.utils.data.TensorDataset(group_1_X, group_1_a, group_1_y)
    group_2_dataset = torch.utils.data.TensorDataset(group_2_X, group_2_a, group_2_y)
    group_1_loader = data.DataLoader(group_1_dataset, batch_size=256, shuffle=False)
    group_2_loader = data.DataLoader(group_2_dataset, batch_size=256, shuffle=False)
    
    x_1 = []
    x_2 = []
    out_1 = []
    out_2 = []
    
    model.eval()
    with torch.no_grad():
        for i, (image, a, y) in enumerate(group_1_loader):
            image = image
            #out, rep, __ = model(image)
            out, rep = model(image)
            x_1.append(rep)
            out_1.append(out)
        for i, (image, a, y) in enumerate(group_2_loader):
            image = image
            #out, rep, __ = model(image)
            out, rep = model(image)
            x_2.append(rep)
            out_2.append(out)
            
    #weight = model.layer_4.weight.data
    weight = model.layer_4.weight.data
    
    
    x_1 = torch.cat(x_1, dim=0)
    x_2 = torch.cat(x_2, dim=0)
    
    #fair_weight_1 = expectation_constrain(weight, x_1, x_2, c=c_1)
    #fair_weight_1 = expectation_constrain(weight, x_1, x_2, c=c_1)
    #fair_weight_2 = covaraince_constrain(fair_weight_1, x_1, x_2, c=c_2)
    #fair_weight_1 = expectation_constrain(weight, x_1, x_2, c=c_1)
    fair_weight_1 = covaraince_constrain(weight, x_1, x_2, c=c_2)
    fair_weight_2 = expectation_constrain(fair_weight_1, x_1, x_2, c=c_1)
    #fair_weight_2 = expectation_constrain(weight, x_1, x_2, c=c_1)
    #fair_weight_1 = expectation_constrain(weight, x_1, x_2, c=c_1)
    #fair_weight_2 = covaraince_constrain(fair_weight_1, x_1, x_2, c=c_2)
    #model.layer_2.weight.data = fair_weight_1
    #model.layer_4.weight.data = fair_weight_2
    model.layer_4.weight.data = fair_weight_2
    
    
    #__, __, rep = model(train_X)
    #weight = torch.matmul(torch.pinverse(torch.matmul(rep.T, rep)), torch.matmul(rep.T, train_y.unsqueeze(1)))
    #model.layer_5.weight.data = weight.T
    
    return model