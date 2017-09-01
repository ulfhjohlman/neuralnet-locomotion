  
  mjpro150
    bin     - dynamic libraries, executables, activation key, MUJOCO_LOG.TXT
    doc     - README.txt and REFERENCE.txt
    include - header files needed to develop with MuJoCo Pro
    model   - model collection (extra models available on the Forum)
    sample  - code samples and makefile need to build them
------------------------------------------------------------------------------------------------------------------------------------------------------------


The distribution contains several header files which are identical on all platforms. They are also available from the links below, to make this documentation self-contained.

mujoco.h   (source)
This is the main header file and must be included in all programs using MuJoCo Pro. It defines all API functions and global variables, and includes the next four files which provide the necessary type definitions.
mjmodel.h   (source)
This file defines the C structure mjModel which is the runtime representation of the model being simulated. It also defines a number of primitive types and other structures needed to define mjModel.
mjdata.h   (source)
This file defines the C structure mjData which is the workspace where all computations read their inputs and write their outputs. It also defines primitive types and other structures needed to define mjData.
mjvisualize.h   (source)
This file defines the primitive types and structures needed by the abstract visualizer.
mjrender.h   (source)
This file defines the primitive types and structures needed by the OpenGL renderer.
mjxmacro.h   (source)
This file is optional and is not included by mujoco.h. It defines X Macros that can automate the mapping of mjModel and mjData into scripting languages, as well as other operations that require accessing all fields of mjModel and mjData. See code sample test.cpp.
glfw3.h
This file is optional and is not included by mujoco.h. It is the only header file needed for the GLFW library. See code sample simulate.cpp.
------------------------------------------------------------------------------------------------------------------------------------------------------------

Naming convention

All symbols defined in the API start with the prefix "mj". The character after "mj" in the prefix determines the family to which the symbol belongs. First we list the prefixes corresponding to type definitions.

mj
Core simulation data structure (C struct), for example mjModel. If all characters after the prefix are capital, for example mjMIN, this is a macro or a symbol (#define).
mjt
Primitive type, for example mjtGeom. Except for mjtByte and mjtNum, all other definitions in this family are enums.
mjf
Callback function type, for example mjfGeneric.
mjv
Data structure related to abstract visualization, for example mjvCamera.
mjr
Data structure related to OpenGL rendering, for example mjrContext.
Next we list the prefixes corresponding to function definitions. Note that function prefixes always end with underscore.

mj_
Core simulation function, for example mj_step. Almost all such functions have pointers to mjModel and mjData as their first two arguments, possibly followed by other arguments. They usually write their outputs to mjData.
mju_
Utility function, for example mju_mulMatVec. These functions are self-contained in the sense that they do not have mjModel and mjData pointers as their arguments.
mjv_
Function related to abstract visualization, for example mjv_updateScene.
mjr_
Function related to OpenGL rendering, for example mjr_render.
mjcb_
Global callback function pointer, for example mjcb_control. The user can install custom callbacks by setting these global pointers to user-defined functions.
------------------------------------------------------------------------------------------------------------------------------------------------------------