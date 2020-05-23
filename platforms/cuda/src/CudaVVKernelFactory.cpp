/* -------------------------------------------------------------------------- *
 *                              OpenMMDrude                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2012 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "CudaVVKernelFactory.h"
#include "CudaVVKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaVVKernelFactory* factory = new CudaVVKernelFactory();
        platform.registerKernelFactory(IntegrateMiddleStepKernel::Name(), factory);
        platform.registerKernelFactory(IntegrateVVStepKernel::Name(), factory);
        platform.registerKernelFactory(ModifyDrudeNoseKernel::Name(), factory);
        platform.registerKernelFactory(ModifyDrudeLangevinKernel::Name(), factory);
        platform.registerKernelFactory(ModifyImageChargeKernel::Name(), factory);
        platform.registerKernelFactory(ModifyElectricFieldKernel::Name(), factory);
        platform.registerKernelFactory(ModifyCosineAccelerateKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerCudaVVKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaVVKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == IntegrateMiddleStepKernel::Name())
        return new CudaIntegrateMiddleStepKernel(name, platform, cu);
    if (name == IntegrateVVStepKernel::Name())
        return new CudaIntegrateVVStepKernel(name, platform, cu);
    if (name == ModifyDrudeNoseKernel::Name())
        return new CudaModifyDrudeNoseKernel(name, platform, cu);
    if (name == ModifyDrudeLangevinKernel::Name())
        return new CudaModifyDrudeLangevinKernel(name, platform, cu);
    if (name == ModifyImageChargeKernel::Name())
        return new CudaModifyImageChargeKernel(name, platform, cu);
    if (name == ModifyElectricFieldKernel::Name())
        return new CudaModifyElectricFieldKernel(name, platform, cu);
    if (name == ModifyCosineAccelerateKernel::Name())
        return new CudaModifyCosineAccelerateKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
