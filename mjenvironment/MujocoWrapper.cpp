#include "MujocoWrapper.h"


mjModel* loadMujocoModel(const char* filename)
{
	// make sure filename is given
	if (!filename)
		throw std::runtime_error("No filename to load.");

	// load and compile
	char error[1000] = "could not load binary model";
	mjModel* mnew = 0;
	if (strlen(filename) > 4 && !strcmp(filename + strlen(filename) - 4, ".mjb"))
		mnew = mj_loadModel(filename, 0);
	else
		mnew = mj_loadXML(filename, 0, error, 1000);
	if (!mnew)
		throw std::runtime_error(error);

	return mnew;
}

std::atomic_int MujocoWrapper::mj_instances = 0;
bool MujocoWrapper::initializeMujoco()
{
	int instances_before = mj_instances.fetch_add(1);
	if (instances_before == 0) {
		int activate_result = mj_activate("mjkey.txt");
		if (activate_result == 0)
			return false;
	}
	return true; //successful
}
