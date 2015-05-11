#include "Packet.h"

Packet::Packet()
{
	size = 0;
	readPosition = 0;
}

Packet::Packet(PacketType type_)
{
	type = type_;
	size = 0;
	readPosition = 0;

	WriteByte((unsigned char)type);
}

unsigned char Packet::ReadByte()
{
	int index = readPosition;
	readPosition += 1;

	if ((size - readPosition) < 0)
		return 0;

	return message[index];
}


unsigned short Packet::ReadShort()
{
	int index = readPosition;
	readPosition += sizeof(short);

	if ((size - readPosition) < 0)
		return 0;	

	return *(unsigned short*)(&message[index]);
}


int Packet::ReadInt()
{
	int index = readPosition;
	readPosition += sizeof(int);

	if ((size - readPosition) < 0)
		return 0;	

	return *(int*)(&message[index]);
}

float Packet::ReadFloat()
{
	int index = readPosition;
	readPosition += sizeof(float);

	if ((size - readPosition) < 0)
		return 0.0f;

	return *(float*)(&message[index]);
}


glm::vec3 Packet::ReadVec3()
{
	int index = readPosition;
	readPosition += sizeof(glm::vec3);

	if ((size - readPosition) < 0)
		return glm::vec3(0.0f);

	return *(glm::vec3*)(&message[index]);
}

glm::quat Packet::ReadQuat()
{
	int index = readPosition;
	readPosition += sizeof(glm::quat);

	if ((size - readPosition) < 0)
		return glm::quat();

	return *(glm::quat*)(&message[index]);
}


bool Packet::ReadBool()
{
	int index = readPosition;
	readPosition += sizeof(bool);

	if ((size - readPosition) < 0)
		return 0;	

	return *(bool*)(&message[index]);
}




bool Packet::WriteByte(unsigned char toWrite)
{
	int index = size;
	size += 1;

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	message[index] = toWrite;

	return true;
}


bool Packet::WriteShort(unsigned short toWrite)
{
	int index = size;
	size += sizeof(short);

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	*(unsigned short*)(&message[index]) = toWrite;

	return true;
}


bool Packet::WriteInt(int toWrite)
{
	int index = size;
	size += sizeof(int);

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	*(int*)(&message[index]) = toWrite;

	return true;
}

bool Packet::WriteFloat(float toWrite)
{
	int index = size;
	size += sizeof(float);

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	*(float*)(&message[index]) = toWrite;

	return true;
}

bool Packet::WriteVec3(glm::vec3 toWrite)
{
	int index = size;
	size += sizeof(glm::vec3);

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	*(glm::vec3*)(&message[index]) = toWrite;

	return true;
}

bool Packet::WriteQuat(glm::quat toWrite)
{
	int index = size;
	size += sizeof(glm::quat);

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	*(glm::quat*)(&message[index]) = toWrite;

	return true;
}

bool Packet::WriteBool(bool toWrite)
{
	int index = size;
	size += sizeof(bool);

	if ((MAX_PACKET_SIZE - size) < 0)
		return false;

	*(bool*)(&message[index]) = toWrite;

	return true;
}

void Packet::WriteCheckSum()
{
	*(int*)(&message[0]) = size;
}