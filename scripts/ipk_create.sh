#!/usr/bin/env bash

# Creates the ipk file to install this project into the SEPP onboard the OPS-SAT spacecraft.

# The project directory path.
# Remove the scripts folder in case this bash script is being executed from the scripts folder
# instead of from the project root folder.
project_dir=$(pwd)
project_dir=${project_dir/scripts/""}

exp_dir=${project_dir}/home/exp1000

# Deployment directory paths.
deploy_dir=${project_dir}/deploy
deploy_home_dir=${deploy_dir}/home
deploy_exp_dir=${deploy_home_dir}/exp1000


# Extract the package name, version, and architecture from the control file.
PKG_NAME=$(sed -n -e '/^Package/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)
PKG_VER=$(sed -n -e '/^Version/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)
PKG_ARCH=$(sed -n -e '/^Architecture/p' ${project_dir}/sepp_package/CONTROL/control | cut -d ' ' -f2)

# Build the ipk filename.
IPK_FILENAME=${PKG_NAME}_${PKG_VER}_${PKG_ARCH}.ipk

# Clean and initialize the deploy folder
rm -rf ${project_dir}/deploy
mkdir ${project_dir}/deploy

# Make a copy of the experiment directory that we will package into an ipk.
# We make a copy because we want to delete some files and folders to not include them in the ipk.
# e.g. We don't want to include local dev binaries that are not compiled for the SEPP's ARM-32 architecture).
cp -a ${project_dir}/home ${deploy_home_dir}

# Can package for the spacecraft (no bash command options) or for the EM (us the 'em' option).
if [ "$1" == "" ]; then         # If packaging for the spacecraft then remove files used for mockin/testing.
    echo "Create IPK for the spacecraft"
    rm -rf ${deploy_exp_dir}/mocks/filestore
    rm -rf ${deploy_exp_dir}/mocks/pictures
    rm -f ${deploy_exp_dir}/config.dev.ini
elif [ "$1" == "em" ]; then     # If not packaging for for the EM.
    echo "Create IPK for the EM"
    rm -f ${deploy_exp_dir}/mocks/filestore/toGround/.gitignore
    rm -f ${deploy_exp_dir}/mocks/filestore/toGround/*.tar.gz
else                            # If not deploying for spacecraft or the EM then an invalid parameter was given.
    echo "Error: invalid option"
    rm -rf ${deploy_dir}
    exit 1
fi

# Remove files that are not needed on the SEPP for both spacecraft and EM deployments.
rm -f ${deploy_exp_dir}/logs/.gitignore
rm -f ${deploy_exp_dir}/requirements.txt
rm -f ${deploy_exp_dir}/bin/armhf/_solib_armhf/.gitignore
rm -f ${deploy_exp_dir}/toGround/.gitignore

# Remove folders that are not needed on the SEPP for both spacecraft and EM deployments.
rm -rf ${deploy_exp_dir}/toGround/*
rm -rf ${deploy_exp_dir}/bin/k8   # Binaries not compiled for the ARM-32 architecture.
rm -rf ${deploy_exp_dir}/kmeans
rm -rf ${deploy_exp_dir}/venv

rm -rf ${deploy_exp_dir}/mocks/__pycache__

# Create the data tar file.
cd ${deploy_dir}
tar -czvf data.tar.gz home --numeric-owner --group=0 --owner=0

# Create the control tar file.
cd ${project_dir}/sepp_package/CONTROL && 
tar -czvf ${deploy_dir}/control.tar.gz control postinst postrm preinst prerm --numeric-owner --group=0 --owner=0
cp debian-binary ${deploy_dir}

# Create the ipk file.
cd ${deploy_dir}
ar rv ${IPK_FILENAME} control.tar.gz data.tar.gz debian-binary
echo "Created ${IPK_FILENAME}"

# Cleanup.
echo "Cleanup"

# Delete the tar files.
rm -f ${deploy_dir}/data.tar.gz
rm -f ${deploy_dir}/control.tar.gz
rm -f ${deploy_dir}/debian-binary

# Delete the home directory.
rm -rf ${deploy_home_dir}

# Done
echo "Done"