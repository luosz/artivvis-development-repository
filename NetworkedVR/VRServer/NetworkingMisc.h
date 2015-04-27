#ifndef NETWORKING_MISC_H
#define NETWORKING_MISC_H

#include "GLM.h"

#define NOTIFICATION_NUMBER 1048

class IPAddress
{
public:
	unsigned int address;
	unsigned short udpPort, tcpPort;


	IPAddress() { }

	IPAddress(unsigned char a, unsigned char b, unsigned char c, unsigned char d, unsigned short port_)
	{
		address = (a << 24) | (b << 16) | (c << 8) | d;
		tcpPort = port_;
	}

	unsigned char A() { return (address >> 24) & (0x00FF); }
	unsigned char B() { return (address >> 16) & (0x00FF); }
	unsigned char C() { return (address >> 8) & (0x00FF); }
	unsigned char D() { return (address & 0x00FF); }
};


#endif