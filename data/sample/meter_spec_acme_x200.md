# ACME X200 Smart Electricity Meter — Technical Specification

## Overview
The ACME X200 is a three-phase smart electricity meter designed for commercial and
light-industrial metering. It supports remote reading over DLMS/COSEM and local
optical port access.

## Electrical Characteristics

| Parameter            | Value                  |
| -------------------- | ---------------------- |
| Reference voltage    | 230/400 V (3-phase)    |
| Voltage range        | 0.8 Un to 1.15 Un      |
| Reference current Ib | 5 A                    |
| Maximum current Imax | 100 A                  |
| Frequency            | 50 Hz                  |
| Accuracy class       | Class 0.5S (active)    |
| Active accuracy      | ±0.5%                  |
| Reactive accuracy    | ±1.0%                  |

## Communications
The X200 communicates using DLMS/COSEM over RS-485 and supports an optional
NB-IoT module. The optical port complies with IEC 62056-21 at 9600 baud.

## Environmental
Operating temperature range is -25 °C to +70 °C. Storage temperature range is
-40 °C to +85 °C. The enclosure protection rating is IP54.

## Pulse Output
The meter provides an LED pulse output of 1000 imp/kWh and an S0 pulse output
configurable between 100 and 10000 imp/kWh.

## Firmware
Firmware is field-upgradable over the air. The current shipping firmware is
version 3.4.2. Upgrades are signed and verified before activation.
