
# Conda environment

Create environment like this:

```txt
conda env create -n <environment_name> -f environment.yml
```

Activate your environment and export the names of its packages like this:

```txt
conda env export --from-history > environment.yml
```

Then export the names of packages along with their versions used on GenomeDK with like this:

```txt
conda env export > environment-genomedk.yml
```
