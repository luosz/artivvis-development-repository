#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "UDPSocket.h"
#include "TCPSocket.h"
#include <vector>
#include <tchar.h>
#include "VolumeDataset.h"

struct Client
{
	IPAddress ipAddress;
	TCPSocket linkedSocket;

	Client() { }

	Client(IPAddress addr, int sockHandle)
	{
		ipAddress = addr;
		linkedSocket.handle = sockHandle;
	}
};

class NetworkManager
{
public:
	VolumeDataset *volume;
	int lastTimestepSent;

	std::vector<Client> clients;

	UDPSocket udpSocket;
	TCPSocket tcpSocket;
	HWND msgWnd;
	
	int udpPort, tcpPort;

	void Init(VolumeDataset &volume_);
	void Update();
	void SendState(int numXBlocks, int numYBlocks, int numZBlocks, int blockRes);
	void SendBlock(int i, int j, int k, int blockRes);

	void InitializeClient(Client &client);

	bool InitTCPMsgHandler();
	void ReadMessage(WPARAM wParam);
	LRESULT CALLBACK  ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
	static LRESULT CALLBACK MsgWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
};


#endif