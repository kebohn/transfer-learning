    #vals_magnitude = torch.linalg.vector_norm(vals, ord=2, dim=1)
    #vals_magnitude_other = torch.linalg.vector_norm(vals_other, ord=2, dim=1)
    #print(vals_magnitude.detach().cpu().reshape(1,-1))
    #print(vals_magnitude_other.detach().cpu().reshape(1,-1))

    #bins = numpy.linspace(min(vals_magnitude_other.detach().cpu()), max(vals_magnitude_other.detach().cpu()), 10)

    #plt.figure()
    #plt.hist(vals_magnitude.detach().cpu().reshape(1,-1), bins, histtype='step', fill=False, label='class')
    #plt.hist(vals_magnitude_other.detach().cpu().reshape(1,-1), bins, histtype='step', fill=False, label='other')
    #plt.savefig(F"hist{idx}.png")