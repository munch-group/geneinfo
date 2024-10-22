gene_lists = {}
for list_name in gene_list_names():
    gene_lists[list_name] = gene_list(list_name)
    
with open('gene_lists.pickle', 'wb') as f:
    pickle.dump(gene_lists, f)

for start in [0, 100000, 200000, 300000, 400000]:
    for step in [500000, 1000000, 1500000, 2000000]:
        print(start, step)
        for p in range(start, 160000000, step):
            plot = gi.gene_plot('chrX', p, p+step, 'hg38')
            plt.close()
            
for start in [0, 100000, 200000, 300000, 400000]:
    for step in [500000, 1000000, 1500000, 2000000]:
        print(start, step)
        for p in range(start, 160000000, step):
            plot = gi.gene_plot('chrX', p, p+step, 'hg19')
            plt.close()            

with open('CACHE.pickle', 'wb') as f:
    pickle.dump(gi.CACHE, f)