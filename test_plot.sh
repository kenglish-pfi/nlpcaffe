while sleep 10; do
    cat a.txt | grep Test | grep category | py --si 'loss = ' 'x[1][:4]' -x | py -c 'matplotlib.use(`Agg`); from matplotlib import pyplot as plt' -C 'plt.savefig("img_test.jpg")' --ji -l 'plt.plot(l)' 2>&1 > /dev/null
done
