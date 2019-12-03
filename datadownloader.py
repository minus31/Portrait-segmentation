import urllib
import glob

URLTXT = "./dataset/alldata_urls.txt"
OUTDIR = "./dataset/images/"

def downloader(URLTXT, OUTDIR):
    with open(URLTXT, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_name = line.split(" ")[0].strip()
            url = line.split(" ")[1].strip()
            if url != "None":
                # Download and write files in dest directory
                img_file = open(OUTDIR + img_name, 'wb')
                img_file.write(urllib.urlopen(url).read())
                img_file.close()
            print(img_name)

downloader(URLTXT, OUTDIR)
