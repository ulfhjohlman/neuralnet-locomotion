#pragma once

// Nonlinear Muscle/torque activation and activation delay
class SomaticMotorNeuronLayer
{
public:
	SomaticMotorNeuronLayer() = default;
	~SomaticMotorNeuronLayer() = default;
	
	SomaticMotorNeuronLayer(const SomaticMotorNeuronLayer& copy_this) = delete;
	SomaticMotorNeuronLayer& operator=(const SomaticMotorNeuronLayer& copy_this) = delete;
	
	SomaticMotorNeuronLayer(SomaticMotorNeuronLayer&& move_this) = delete;
	SomaticMotorNeuronLayer& operator=(SomaticMotorNeuronLayer&& move_this) = delete;
	
private:
	
};