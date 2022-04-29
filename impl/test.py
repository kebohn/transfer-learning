import os

for car in os.listdir("/local/scratch/bohn/datasets/virus/train/"):
  if os.path.isdir(F"/local/scratch/bohn/datasets//virus/train/{car}"):
    size = len([f for f in os.listdir(F"/local/scratch/bohn/datasets/virus/train/{car}")])
    print(F"{car}: {size}")