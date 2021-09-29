import sys
myargv = [i for i in range(1,len(sys.argv))]
myargs =  [sys.argv[i] for i in myargv if i %2 ==1]
myvars = [sys.argv[i] for i in myargv if i %2 ==0]
commands = ['-i','-v','-h', '-dbs','-r1','-r2','-L']
#make args into a dictionary
d = {}
if '-h' in str(sys.argv):
    read = open('ReadMe.md','r').readlines()
    for i in read:
        print(i)
else:
    try:
        for i in range(0,len(myargs)):
            d[myargs[i]]=myvars[i]
    except:
        try:
            print('No var for with %s'%myargs[i])
        except:
            print('No arg for with %s'%myvars[i])

#make sure that the commands are clean
pas = True
if d!={}:
    if '-dbs' not in d.keys():
        print('-dbs is a required command \n use -h for help\n')
        pas = False
    if '-i' not in d.keys():
        print('-i is a required command \n use -h for help\n')
        pas = False

    for i in d.keys():
        if i not in commands:
            print('%s is not a command \n use -h for help\n'%i)
            pas = False

    #check the paths
if pas == True:
    if d['-i']=='0':
        print('Assembly via Unicycler -r1 %s r2 %s -L %s'%(d['-r1'],d['-r2'],d['-L']))
    if d['-i']=='1':
        print('Using Assembled contigs %s'%d['-fasta'])
