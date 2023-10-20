# Read list of files containing radius and positions of spherules from aggregate_generator
# Create input files for MSTM

import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random
from os import listdir
from os.path import isfile, join


def write_mstm_inp(pos_file,in_file,out_file,n_spheres="10",length_scale_factor="1.d0",
                   ref_idx_R="1.4d0",ref_idx_Im="0.4d0",mie_eps="1d-6",batchn=0,printfile="printfile.dat",
                   tmname="tmdefault.dat"):
    dirname = "mstm_v3.0"
    path = os.path.join(dirname, in_file)

    file1 = open(path,'w')
    file1.write("number_spheres\n")
    file1.write(n_spheres+"\n")

    # File containing sphere size, position [radius, X, Y, Z] positions of the ith sphere
    # Units are arbitrary (but should be consistent for radius and position)
    file1.write("sphere_position_file\n")
    file1.write("../aggregate_generator/output_new/"+pos_file+"\n")

    file1.write("output_file\n")
    file1.write("out"+str(int(batchn))+"/"+out_file+"\n")

    file1.write("run_print_file\n")
    file1.write(printfile+"\n")

    # Size parameter is length_scale_factor*radius of ith sphere
    # scale factor: default is 1.0
    file1.write("length_scale_factor\n")
    file1.write(length_scale_factor+"\n")

    # Real part of refractive index of all spheres (or factor multiplied by value given in sphere position file)
    file1.write("real_ref_index_scale_factor\n")
    file1.write(ref_idx_R+"\n")

    #Imag. part of refractive index of all spheres (or factor multiplied by value given in sphere position file)
    file1.write("imag_ref_index_scale_factor\n")
    file1.write(ref_idx_Im+"\n")

#   file1.write("mie_epsilon\n")
#    file1.write(mie_eps+"\n")
    file1.write("normalize_scattering_matrix\n")
    file1.write("1\n")

    file1.write("calculate_scattering_coefficients\n")
    file1.write("1\n")

    file1.write("min_scattering_angle_deg\n")
    file1.write("0.0d0\n")

    file1.write("max_scattering_angle_deg\n")
    file1.write("180.d0\n")

    file1.write("min_scattering_plane_angle_deg\n")
    file1.write("0.0d0\n")

    file1.write("max_scattering_plane_angle_deg\n")
    file1.write("0.0d0\n")

    file1.write("delta_scattering_angle_deg\n")
    file1.write("1\n")

    # 0 - fixed orientation (default), 1 - random orientation
    file1.write("fixed_or_random_orientation\n")
    file1.write("1\n")

    # 0 - only total properties written to output, 1 - input/calculated individual sphere properties written (default)
    file1.write("write_sphere_data\n")
    file1.write("1\n")

    file1.write("t_matrix_file\n")
    file1.write(tmname+"\n")

    file1.write("sm_number_processors\n")
    file1.write("24\n")

    #file1.write("new_run\n")

    #file1.write("multiple_run\n")
    #file1.write("length_scale_factor\n")
    #file1.write("0.1d0 0.7d0 0.2d0\n")
    file1.write("end_of_options\n")

    file1.close()
    print(path)

def write_batch_script(filenames,valNs,valDfs):
    dirname = "mstm_v3.0"

    Dfs = ["1.8", "1.9", "2.0", "2.1", "2.2","2.3"]
    print(valNs[0:10])
    print(valDfs[0:10])

    # sfilenames=[x for x,_ in sorted(zip(filenames,valNs))]
    # svalDfs= [x for x,_ in sorted(zip(valDfs,valNs))]

    for j in range(len(Dfs)):
        path = os.path.join(dirname, "mstm_run_f"+str(j)+".sh")

        file2 = open(path,'w')
        file2.write("#!/bin/sh\n")
        file2.write("##\n")
        file2.write("#SBATCH --account=ACCT\n")
        file2.write("#SBATCH --job-name=mstm_run"+str(j)+"\n")
        file2.write("#SBATCH -n 24\n")
        file2.write("#SBATCH --time=11:59:00\n")
        file2.write("#SBATCH --mem=128gb\n")
        file2.write("#SBATCH --mail-type=ALL\n")
        file2.write("#SBATCH --mail-user=emailaddress\n")
        file2.write("\n")
        file2.write("module load intel-parallel-studio/2017\n")

        for ii in range(len(filenames)):
            if (Dfs[j]==valDfs[ii]):
                file2.write("mpirun -np 24 ./mstm_mpif90.out "+filenames[ii]+"\n")
                file2.write("date\n")

        file2.write("\n")
        file2.write("# End of script\n")
        file2.close()
        print(path)

def write_batch_script_nsplit(filenames,batchnums,valNs,valaggs,alreadyoutput):
    dirname = "mstm_v3.0"

    nsplits = 10.0 #10.0
    print(nsplits)

    # sfilenames=[x for x,_ in sorted(zip(filenames,valNs))]
    # svalDfs= [x for x,_ in sorted(zip(valDfs,valNs))]

    for j in range(int(nsplits)):
        path = os.path.join(dirname, "mstm_run_all"+str(j)+".sh")

        file2 = open(path,'w')
        file2.write("#!/bin/sh\n")
        file2.write("##\n")
        file2.write("#SBATCH --account=ACCT\n")
        file2.write("#SBATCH --job-name=mstm_run"+str(j)+"\n")
        file2.write("#SBATCH -N 1\n")
        file2.write("#SBATCH -n 24\n")
        file2.write("#SBATCH --time=11:59:00\n")
        file2.write("#SBATCH --mail-type=ALL\n")
        file2.write("#SBATCH --mail-user=emailaddress\n")
        file2.write("\n")
        file2.write("module load intel-parallel-studio/2017\n")

        for ii in range(len(filenames)):
            Nscurr=float(valNs[ii])
            if ((batchnums[ii]==int(j))and(Nscurr>400)and(Nscurr<500)and(valaggs[ii]<10)and(alreadyoutput[ii]==False)):
                #print(Nscurr)
                if (Nscurr < 30.0):
                    file2.write("./mstm"+str(int(batchnums[ii]))+"/mstm_mpif90.out "+filenames[ii]+"\n")
                elif (Nscurr <50.0):
                    file2.write("mpirun -np 4 ./mstm" + str(int(batchnums[ii])) + "/mstm_mpif90.out "+filenames[ii]+"\n")
                elif (Nscurr <81.0):
                    file2.write("mpirun -np 10 ./mstm" + str(int(batchnums[ii])) + "/mstm_mpif90.out "+filenames[ii]+"\n")
                else:
                    file2.write("mpirun -np 24 ./mstm" + str(int(batchnums[ii])) + "/mstm_mpif90.out "+filenames[ii]+"\n")

                file2.write("wait\n")
                file2.write("date\n")
        file2.write("\n")
        file2.write("# End of script\n")
        file2.close()
        print(path)
def calc_nn_array(num_sph_SA,n_levels):
    # for the aggregate_gen code return size of aggregates for given parameters
    ss = np.asarray([2**i for i in range(n_levels+1)])
    print(num_sph_SA*ss)
    return num_sph_SA*ss
def makenndict(Ns,n_levels,batchnum):
    NN=calc_nn_array(Ns,n_levels)
    nndict={}
    for i in NN:
        nndict[i] = batchnum
    return nndict
def read_aggregator_gen(strfilename):
    #strlist=strfilename.split("_")
    m = re.match("agg(.*)_N(.*)_kf(.*)_Df(.*).out",strfilename)
    #print(m.group(1),m.group(2),m.group(3),m.group(4))

    values = [m.group(1),m.group(2),m.group(3),m.group(4)]

    return values
def sort_aggfiles(aggfiles):
    #sorts aggfiles by Ns size
    def return_ns(ii):
        vals = read_aggregator_gen(ii)
        ns = int(vals[1])
        return ns

    sortedaggfiles = sorted(aggfiles, key=return_ns)

    return sortedaggfiles
def read_mstm_out(strfilename):
    if (strfilename[-1]=="f"):
        m = re.match("mstm0_N(.*)_kf(.*)_Df(.*)f", strfilename)
    else:
        m = re.match("mstm0_N(.*)_kf(.*)_Df(.*)", strfilename)
    print(m)
    print(m.group(1), m.group(2), m.group(3))

    values = [m.group(1), m.group(2), m.group(3)]

    return values

if __name__ == "__main__":

    # path to directory with the aggregate_generator files
    aggdir = "aggregate_generator/output_new/"
    origaggfiles = [f for f in listdir(aggdir) if isfile(join(aggdir, f))]
    aggfiles = sort_aggfiles(origaggfiles)
    print(len(aggfiles))

    # Files that already have output
    mstmdir = "mstm_v3.0/"
    outfiles = ["out.txt", "out0.txt", "out1.txt", "out2.txt", "out3.txt", "out4.txt", "out5.txt", "out6.txt",
                "out7.txt","out8.txt", "out9.txt"]
    output = []
    sizes = []
    for oo in outfiles:
        fn = os.path.join(mstmdir, oo)
        print(fn)
        lahtr = pd.read_csv(fn, header=None, delim_whitespace=True)
        size = lahtr[4]
        names = lahtr[8]
        output.append(names.values.tolist())
        sizes.append(size.values.tolist())

    alloutput = [item for sublist in output for item in sublist]
    allsizes = [item for sublist in sizes for item in sublist]

    npsizes = np.zeros(len(allsizes))
    for i in range(len(allsizes)):
        npsizes[i] = float(allsizes[i][:-1])

    # only include the ones with reasonable output file size
    goodoutput = []
    for j in range(len(alloutput)):
        if npsizes[j] > 150:
            goodoutput.append(alloutput[j])
    print(len(goodoutput))
     # Range of microphysical parameters for BC [from Liu et al. 2018]
    ##### Needed for Aggregate generator
    # Aggregates by clusters - needed to create keys for batch number by cluster size
    n0dict = makenndict(3, 10, 0)
    n1dict = makenndict(4, 9, 1)
    n2dict = makenndict(5, 9, 2)
    n3dict = makenndict(7, 8, 3)
    n4dict = makenndict(9, 8, 4)
    n5dict = makenndict(11, 8, 5)
    n6dict = makenndict(13, 8, 6)
    n7dict = makenndict(15, 7, 7)
    n8dict = makenndict(17, 7, 8)
    n9dict = makenndict(19, 7, 9)

    dicts = [n0dict, n1dict, n2dict, n3dict, n4dict, n5dict, n6dict, n7dict, n8dict, n9dict]
    nsdict = {}
    for k in set(k for d in dicts for k in d):
        nsdict[k] = [d[k] for d in dicts if k in d]

    print(len(nsdict))

    # range of fractal prefactors
    kfs = [1.2]
    # range of fractal dimensions
    Dfs = np.linspace(1.8,2.3,num=5)

    #### Needed for MSTM calculation
    # range of size parameters
    aas = np.linspace(0.1,0.7,num=4)
    #aas = [0.3 0.5 0.7 0.9]
    # range of index of refraction
    reals = np.linspace(1.4,2.0,num=4)
    realnames = ["14","16","18","20"]
    imags = np.linspace(0.4,1.0,num=4)

    vals = []
    valstr = []
    valNs = []
    valDfs = []
    outfiles = []
    batches = []
    alreadyoutput = []

    lsfvalid = [0,1,2,3]
    lsfvals10 = ["5", "7", "9", "10"]
    lsfvals = ["0.5", "0.7", "0.9", "1.0"]


    for ff in aggfiles:
        values=read_aggregator_gen(ff)
    #for ff in mstmnooutfiles:
        print(ff)
        #values = read_mstm_out(ff)
        batchnum=nsdict.get(int(values[1]))
        if batchnum is not None:
            whichbatch = batchnum[0]
        else:
            print("*****NOT A VALID NS: ",ff)
            whichbatch = 10

        for jj in range(len(reals)):
            realstr=format(reals[jj],'0.1f')+"d0"
            imgstr=format(imags[jj],'0.1f')+"d0"
            print(whichbatch,realstr,imgstr)
            lsf = random.choice(lsfvalid)#"3"
            lsfstr = "{}d0".format(lsfvals[lsf])
            mstm_inp_filename = "in"+str(whichbatch)+"/a" + values[0] + "N" + values[1] + "R" + realnames[jj] + "Df" + values[3]+"x"+lsfvals10[lsf]+".inp"
            mstm_out_filename = "a" + values[0] + "N" + values[1] + "R" + realnames[jj] + "Df" + values[3] + "x"+lsfvals10[lsf]+".out"
            tmfile = "tmatrix"+str(whichbatch)+".dat"
            prfile = "print"+str(whichbatch)+".dat"
            outfiles.append(mstm_inp_filename)
            batches.append(whichbatch)
            vals.append([float(ii) for ii in values])
            valstr.append([ii for ii in values])

            print(mstm_inp_filename,mstm_out_filename)

            if mstm_out_filename in goodoutput:
                written = True
            else:
                written = False
            write_mstm_inp(ff,
                    mstm_inp_filename,
                    mstm_out_filename,
                    n_spheres=values[1],
                    length_scale_factor=lsfstr,
                    ref_idx_R=realstr,
                    ref_idx_Im=imgstr,
                    mie_eps="1d-6",
                    batchn=whichbatch,
                    printfile=prfile,
                    tmname=tmfile)
            alreadyoutput.append(written)
            print(written)
    valnp=np.array(vals)

    valstrnp = np.array(valstr)
    print(valnp.shape)

    valaggs =valnp[:,0].tolist()
    valNs = valnp[:,1].tolist()
    valDfs = valstrnp[:,3].tolist()
    print("Start writing batch scripts")
    print(len(valNs))
    # write the batch file to run on habanero
    write_batch_script_nsplit(outfiles,batches,valNs,valaggs,alreadyoutput)

    plt.scatter(valnp[:,1],valnp[:,3])
    plt.xticks(np.arange(0,max(valnp[:,1]),500))
    plt.show()

    plt.scatter(valnp[:,1],valnp[:,3])
    plt.xticks(np.arange(0,max(valnp[:,1]),500))
    plt.show()
