pyplt() {
  py -c 'matplotlib.use(`Agg`); from matplotlib import pyplot as plt' -C 'plt.savefig("output.jpg")'
}
while sleep 10; do
    cat a.txt | grep test | grep category | py --si 'loss = ' 'x[1][:4]' -x | pyplt --ji -l 'plt.plot(l)'
done
