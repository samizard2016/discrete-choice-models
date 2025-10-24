from __future__ import annotations
import pickle
import os
import wmi
from typing import List
import pprint as pp
from datetime import datetime
import json
import logging
class Registry:
    def __init__(self,**kwargs) -> None:
        self.logger = logging.getLogger("kdcm_logger")
        self.d_mac = {"expired":"no"}
        self.d_mac.update(kwargs)  
        self.save()     
    def save(self) -> None:
        try:
            if 'machine_name' not in self.d_mac and 'hdd' not in self.d_mac:
                self.d_mac["machine_name"] =  os.environ['COMPUTERNAME']
                self.d_mac['hdd'] = self.hdd_serial()
            d_bin = Registry.dict_to_binary(self.d_mac)           
            reg_file = f"{self.d_mac['machine_name']}.bin"
            with open(f"{reg_file}", 'wb') as handle:
                pickle.dump(d_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)           
        except Exception as err:
            self.logger.error(f"Couldn't save the registration file {reg_file}:{err}")
    def update(self):
        reg_file = f"{self.d_mac['machine_name']}.bin"
        d_bin = Registry.dict_to_binary(self.d_mac)
        with open(f"{reg_file}", 'wb') as handle:
            pickle.dump(d_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def restore(reg_file) -> Registry:
        with open(reg_file, 'rb') as handle:
            reg = pickle.load(handle)
            reg = Registry.binary_to_dict(reg)
            return Registry(**reg)
    def hdd_serial(self) -> str:
        c = wmi.WMI()
        for item in c.Win32_PhysicalMedia():
            if "PHYSICALDRIVE" in str(item.Tag).upper():
                serialNo = item.SerialNumber
                return str(serialNo)
        self.logger.info(self)
    def __str__(self):
        return f"{self.d_mac['machine_name']}:\
            {self.d_mac['hdd'].strip()}:\
            {self.d_mac['expiry_date']}"
    
    def xcode(self,x: str) -> dict:
        self.d_mac['xcode'] = x
        return self.d_mac
    def check_xcode(self) -> bool:
        if "xcode" in self.d_mac:
            return self.d_mac['xcode']=="346B"
        else:
            return False
    def check_exp_date(self):
        exp_date = datetime.strptime(self.d_mac['expiry_date'], '%Y-%m-%d').date()
        if exp_date > datetime.now().date():
            return True
        else:
            return False
    def check(self) -> bool:
        if self.check_xcode() and self.d_mac['expired'] != "yes" and\
           self.d_mac['hdd']==self.hdd_serial() and self.check_exp_date():
           return True
        else:
            self.logger.warning(self.d_mac)
            return False
    @staticmethod
    def dict_to_binary(the_dict):
        str = json.dumps(the_dict)
        binary = ' '.join(format(ord(letter), 'b') for letter in str)
        return binary
    @staticmethod
    def binary_to_dict(the_binary):
        try:
            jsn = ''.join(chr(int(x, 2)) for x in the_binary.split())
            d = json.loads(jsn)  
            return d
        except Exception as err:
            self.logger.error(f"failed in converting d_mac into binary: {err}")
if __name__=="__main__":
    # reg = Registry(**{"expiry_date":"2023-03-31"})
    # reg.xcode("346B")
    # print(reg)
    # print(reg.check())
    # reg.update()
    # restore from user reg file and update as follows
    # and share dexreg.bin
    d = Registry.restore('AAD-PF4NK928.bin')
    print(d)
    d.xcode("346B")
    d.d_mac["expiry_date"] = "2026-12-31"
    d.d_mac["expired"] = "no"
    reg = Registry(**d.d_mac)
    reg.update()
    print(reg.check_xcode())
    print(reg.check_exp_date())
    print(reg.d_mac)