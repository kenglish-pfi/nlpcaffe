while sleep 10; do
    for d in `find .. -maxdepth 1 -type d -name '*caffe*'`; do
        cd $d
        cat a.txt | grep Test | grep -v Iteration | py --si 'loss = ' 'x[1][:4]' -x | py -l --ji 'numpy.array(l[::2]) + numpy.array(l[1::2])' | py -c 'matplotlib.use(`Agg`); from matplotlib import pyplot as plt' -C 'plt.savefig("img_test.jpg")' --ji -l 'plt.plot(l)' 2>&1 > /dev/null
    done
done
while sleep 10; do
