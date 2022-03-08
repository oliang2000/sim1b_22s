import os
npixel = 37 #about 10 arcsec

if not os.path.isdir("./stamp"):
    os.system("mkdir ./stamp")

if not os.path.isdir("./stamp/fits/"):
    os.system("mkdir ./stamp/fits/")
    
if not os.path.isdir("./stamp/png/"):
    os.system("mkdir ./stamp/png/")

def DownloadByRaDec_PNG(ra,dec):
    save_dir = "./stamp/png/"

    filename = "ra_"+str(ra)+"_dec_"+str(dec)+".png"

    
    cmd1 = "wget -P " + save_dir + \
    " 'http://legacysurvey.org/viewer/jpeg-cutout?ra="+str(ra)+"&dec="+str(dec)+"&width="+str(npixel)+"&height="+str(npixel)+"&pixscale=0.27&layer=dr8&bands=grz\'"
    cmd2 = "mv "+ save_dir + "jpeg-cutout* " + save_dir + filename

    os.system(cmd1)
    os.system(cmd2)

def DownloadByRaDec_FITS(ra,dec):
    save_dir = "./stamp/fits/"

    filename = "ra_"+str(ra)+"_dec_"+str(dec)+".fits"

    
    cmd1 = "wget -P " + save_dir + \
    " 'http://legacysurvey.org/viewer/fits-cutout?ra="+str(ra)+"&dec="+str(dec)+"&width="+str(npixel)+"&height="+str(npixel)+"&pixscale=0.27&layer=dr8&bands=grz\'"
    cmd2 = "mv "+ save_dir + "fits-cutout* " + save_dir + filename

    os.system(cmd1)
    os.system(cmd2)
