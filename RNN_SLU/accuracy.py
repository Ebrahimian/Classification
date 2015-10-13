
def accuracy(p,g,w,fileName):

    correct=0
    total=0
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):

            out += w + ' ' + wl + ' ' + wp + '\n'
            total+=1
            if wl==wp: correct+=1
        out += 'EOS O O\n\n'
    acc=(correct/float(total))*100
    f = open(fileName,'w')
    f.writelines(out)
    f.close()

    return(acc,acc,acc)
