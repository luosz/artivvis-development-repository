#ifndef PACKET_H
#define PACKET_H

#include "GLM.h"

#define MAX_PACKET_SIZE 60000
//ideal 1.4kb, just under MTU

enum class PacketType { INITIALIZATION, BLOCK };

class Packet
{
public:
	int size;
	int readPosition;
	unsigned char message[MAX_PACKET_SIZE];
	//#pragma align 32bit
	
	PacketType type;

	Packet();
	Packet(PacketType type_);

	unsigned char ReadByte();
	unsigned short ReadShort();
	int ReadInt();
	float ReadFloat();
	glm::vec3 ReadVec3();
	glm::quat ReadQuat();
	bool ReadBool();

	bool WriteByte(unsigned char toWrite);
	bool WriteShort(unsigned short toWrite);
	bool WriteInt(int toWrite);
	bool WriteFloat(float toWrite);
	bool WriteVec3(glm::vec3 toWrite);
	bool WriteQuat(glm::quat toWrite);
	bool WriteBool(bool toWrite);

	void WriteCheckSum();
};

#endif