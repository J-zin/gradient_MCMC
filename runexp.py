import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.animation import FuncAnimation

from model.sgvd import SVGD
from model.langevin import LD
from model.guassians import Gaussian_dist_1D, Mix_Gaussian_dist_1D, Gaussian_dist_2D, Mix_Gaussian_dist_2D


def particles_visual_1dgaussian(model_name='svgd'):
    # draw 1d gaussian
    x = np.linspace(-12, 12, 300)
    model = Mix_Gaussian_dist_1D(Gaussian_dist_1D(-2, 1), Gaussian_dist_1D(2, 1), 1/3, 2/3)
    prob_x = np.exp(model.logprob(x))
    fig = plt.figure()
    plt.grid(False)
    plt.xlim([-12, 12])
    plt.plot(x, prob_x, color='r', lw=5, ls='--')

    # particles visualization
    init_x = np.random.normal(-10, 1, [10, 1])
    if model_name == 'svgd':
        theta_list = SVGD().update(init_x, model.dlogprob, n_iter=2000, stepsize=0.01)
        plt.xlabel('SVGD', fontsize = 20, fontweight="bold")
    elif model_name == 'ld':
        theta_list = LD().update(init_x, model.dlogprob, n_iter=2000, stepsize=0.1, adagrad=False)
        plt.xlabel('Langiven Dynamics', fontsize = 20, fontweight="bold")
    else: exit()
    prob_list = np.exp(model.logprob(theta_list))

    c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    scat = plt.scatter(theta_list[0], prob_list[0], s=100, c=c_list)
    title = plt.suptitle(t='', fontsize = 20, fontweight="bold")
    def animate(i):
        X = theta_list[i]
        Y = prob_list[i]
        scat.set_offsets(np.c_[X, Y])
        title.set_text('{}th Iteration'.format(i))

    anim = FuncAnimation(fig, animate, interval=1, frames=len(theta_list)-1, repeat=True)
    print("approximate mean: ", np.mean(theta_list[-1]))
    print("true mean: ", -1/3 * 2 + 2/3 * 2)
    plt.show()
    # anim.save('1dld.gif', writer='pillow', fps=50)

def kde_visual_1dgaussian(model_name='svgd'):
    # draw 1d gaussian
    x = np.linspace(-12, 12, 300)
    model = Mix_Gaussian_dist_1D(Gaussian_dist_1D(-2, 1), Gaussian_dist_1D(2, 1), 1/3, 2/3)
    prob_x = np.exp(model.logprob(x))
    fig = plt.figure()
    plt.grid(False)
    plt.xlim([-12, 12])
    plt.plot(x, prob_x, color='r', lw=5, ls='--')

    # kernel density estimator visualization
    init_x = np.random.normal(-10, 1, [100, 1])     # 100 particles
    theta_list_svgd = SVGD().update(init_x, model.dlogprob, n_iter=2000, stepsize=0.01)
    theta_list_ld = LD().update(init_x, model.dlogprob, n_iter=2000, stepsize=0.1, adagrad=False)

    prob_list_svgd = []
    prob_list_ld = []
    for i in range(len(theta_list_svgd)):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(theta_list_svgd[i].reshape(-1,1))
        prob_x = np.exp(kde.score_samples(x.reshape(-1,1)))
        prob_list_svgd.append(prob_x)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(theta_list_ld[i].reshape(-1,1))
        prob_x = np.exp(kde.score_samples(x.reshape(-1,1)))
        prob_list_ld.append(prob_x)

    title = plt.suptitle(t='', fontsize = 20, fontweight="bold")
    line_svgd, = plt.plot(x, prob_list_svgd[0], color='g', lw=5, ls='-', label='SVGD')
    line_ld, = plt.plot(x, prob_list_ld[0], color='b', lw=5, ls='-', label='Langevin Dynamics')
    plt.legend()
    def animate(i):
        title.set_text('{}th Iteration'.format(i))
        line_svgd.set_data(x, prob_list_svgd[i])
        line_ld.set_data(x, prob_list_ld[i])
    anim = FuncAnimation(fig, animate, interval=1, frames=len(theta_list_svgd)-1, repeat=True)
    plt.show()
    # anim.save('kde.gif', writer='pillow', fps=50)

def particles_visual_2dgaussian(model_name='svgd'):
    mesh = []
    grid_size = 100
    x = np.linspace(-8, 8, grid_size)
    y = np.linspace(-8, 8, grid_size)
    for i in x:
            for j in y:
                mesh.append(np.asarray([i, j]))
    mesh = np.stack(mesh, axis=0)

    model = Mix_Gaussian_dist_2D(Gaussian_dist_2D([5, 5], [1,1]), Gaussian_dist_2D([-5, -5], [1,1]), 0.8, 0.2)
    prob = np.reshape(np.exp(model.logprob(mesh)), [grid_size, grid_size])

    fig = plt.figure()
    plt.grid(False)
    plt.axis('off')
    plt.axis('square')
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.imshow(np.flipud(prob), cmap='YlOrRd', extent=[-8, 8, -8, 8])

    grid_size = 20
    mesh = []
    x = np.linspace(-8, 8, grid_size)
    y = np.linspace(-8, 8, grid_size)
    for i in x:
        for j in y:
            mesh.append(np.asarray([i, j]))
    mesh = np.stack(mesh, axis=0)
    scores = model.dlogprob(mesh)
    plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=0.005)

    init_x = np.random.uniform(0, 1, [1000, 2]) * 16 - 8
    if model_name == 'svgd':
        sample_list = SVGD().update(init_x, model.dlogprob, n_iter=500, stepsize=0.1, adagrad=False)
        plt.xlabel('SVGD', fontsize = 20, fontweight="bold")
    elif model_name == 'ld':
        sample_list = LD().update(init_x, model.dlogprob, n_iter=500, stepsize=0.1, adagrad=False)
        plt.xlabel('Langiven Dynamics', fontsize = 20, fontweight="bold")
    else: exit()

    title = plt.suptitle(t='', fontsize = 20, fontweight="bold")
    scat = plt.scatter(sample_list[0][:, 0], sample_list[0][:, 1], s=0.5)
    def animate(i):
        X = sample_list[i][:, 0]
        Y = sample_list[i][:, 1]
        scat.set_offsets(np.c_[X, Y])
        title.set_text('{}th Iteration '.format(i) + ('(SVGD)' if model_name == 'svgd' else '(LD)'))

    anim = FuncAnimation(fig, animate, interval=1, frames=len(sample_list)-1, repeat=True)

    plt.show()
    # anim.save('2dld.gif', writer='pillow', fps=50)


if __name__ == '__main__':
    particles_visual_1dgaussian()
    # kde_visual_1dgaussian()
    # particles_visual_2dgaussian()