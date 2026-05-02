#include <Arduino.h>
#include <MPR121.h>
#include <Streaming.h>

#include "Constants.h"

// Initialize the MPR121 object
MPR121 mpr121;

uint16_t initial_baselines[5];

void setup()
{
  // Start serial communication at the speed defined in Constants.h
  Serial.begin(constants::baud);
  Serial.println("MPR121 Analog Field Reader Initialized");

  // Setup the MPR121 sensor using all the settings from your constants files
  mpr121.setupSingleDevice(*constants::wire_ptr,
    constants::device_address,
    constants::fast_mode);

  mpr121.setAllChannelsThresholds(constants::touch_threshold,
    constants::release_threshold);
  mpr121.setDebounce(constants::device_address,
    constants::touch_debounce,
    constants::release_debounce);
  mpr121.setChargeDischargeCurrent(constants::device_address,
    constants::charge_discharge_current);
  mpr121.setChargeDischargeTime(constants::device_address,
    constants::charge_discharge_time);
  mpr121.setFirstFilterIterations(constants::device_address,
    constants::first_filter_iterations);
  mpr121.setSecondFilterIterations(constants::device_address,
    constants::second_filter_iterations);
  mpr121.setSamplePeriod(constants::device_address,
    constants::sample_period);
  
  // Set the baseline initially and then lock it in the loop.
    mpr121.setBaselineTracking(constants::device_address, constants::baseline_tracking);
  // Start the sensor so it begins taking readings
  mpr121.startChannels(constants::physical_channel_count,
    constants::proximity_mode);
  
  Serial.println("Calibrating baseline... Please keep hands away!");
  delay(constants::setupTime);

  //lock the baseline after calibration
  mpr121.setBaselineTracking(constants::device_address, MPR121::BASELINE_TRACKING_DISABLED);

  //fir the initial reading for baseline
  for(int i = 0; i< constants::physical_channel_count; i++){
    initial_baselines[i] = mpr121.getChannelBaselineData(i);
    Serial.print("Channel ");
    Serial.print(i);
    Serial.print("Initial baseline: ");
    Serial.print(initial_baselines[i]);
  }
  Serial.println();

  Serial.println("Setup complete Starting Now.....");
}

void loop()
{
  // Check if the sensor is connected and communicating properly
  if (!mpr121.communicating(constants::device_address))
  {
    Serial.println("MPR121 device not communicating!");
    delay(constants::loop_delay); // Wait a second before trying again
    return;
  }

  // Loop through all 5 channels (0 to 4)
  // for (int i = 0; i < constants::physical_channel_count; i++)
  // {
    // Get the filtered (analog) data from the current channel in the loop
    // uint16_t field_plate_4 = mpr121.getDeviceChannelFilteredData(constants::device_address, 4);
    // uint16_t raw_values =  963; //Manually_tuned
    // // uint16_t raw_values = mpr121.getDeviceChannelBaselineData(constants::device_address,4);
    // int16_t error = raw_values-field_plate_4;
  for (int i = 0; i < constants::physical_channel_count; i++)
  {
    uint16_t field = mpr121.getChannelFilteredData(i);
    // uint16_t raw_values =  963; //Manually_tuned
    uint16_t raw_values = mpr121.getChannelBaselineData(i);
    // int16_t error2 = raw_values - field;
    int16_t error = raw_values-field;

    // Print this value to the Serial Monitor, indicating which channel it is
    Serial.print(" Channel ");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(field);
    // Serial.print(", Raw :");
    // Serial.print(raw_values);
    Serial.print(", error :");
    Serial.print(error);
  }
  Serial.println();


  // A short delay so the Serial Monitor is not flooded with data
  delay(constants::print_delay);
}
