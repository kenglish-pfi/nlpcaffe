cat a.txt | grep Iteration | grep loss | py --si 'loss = ' 'x[1][:4]' -x | py -l l[10:] | py --ji -l 'numpy.convolve(numpy.ones(100)/100, l)' | py --ji -l l[100:-100] | pyplt --ji -l 'plt.plot(l)'
