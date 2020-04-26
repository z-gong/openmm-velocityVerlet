/* -------------------------------------------------------------------------- *
 *                                OpenMMDrude                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/DrudeForce.h"
#include "openmm/VVIntegrator.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace OpenMM;
using namespace std;

extern "C" void registerVelocityVerletSerializationProxies();

void testSerialization() {
    // Create an Integrator.

    VVIntegrator integ1(300.0, 10.0, 1.0, 40.0, 0.001, 3, 1);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<VVIntegrator>(&integ1, "Integrator", buffer);
    VVIntegrator* copy = XmlSerializer::deserialize<VVIntegrator>(buffer);

    // Compare the two integrators to see if they are identical.

    VVIntegrator& integ2 = *copy;
    ASSERT_EQUAL(integ1.getTemperature(), integ2.getTemperature());
    ASSERT_EQUAL(integ1.getFrequency(), integ2.getFrequency());
    ASSERT_EQUAL(integ1.getDrudeTemperature(), integ2.getDrudeTemperature());
    ASSERT_EQUAL(integ1.getDrudeFrequency(), integ2.getDrudeFrequency());
    ASSERT_EQUAL(integ1.getStepSize(), integ2.getStepSize());
    ASSERT_EQUAL(integ1.getNumNHChains(), integ2.getNumNHChains());
    ASSERT_EQUAL(integ1.getLoopsPerStep(), integ2.getLoopsPerStep());
    ASSERT_EQUAL(integ1.getUseCOMTempGroup(), integ2.getUseCOMTempGroup());

    ASSERT_EQUAL(integ1.getConstraintTolerance(), integ2.getConstraintTolerance());
    ASSERT_EQUAL(integ1.getMaxDrudeDistance(), integ2.getMaxDrudeDistance());

    ASSERT_EQUAL(integ1.getFriction(), integ2.getFriction());
    ASSERT_EQUAL(integ1.getDrudeFriction(), integ2.getDrudeFriction());
    ASSERT_EQUAL(integ1.getRandomNumberSeed(), integ2.getRandomNumberSeed());

    ASSERT_EQUAL(integ1.getMirrorLocation(), integ2.getMirrorLocation());
}

int main() {
    try {
        registerVelocityVerletSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}

