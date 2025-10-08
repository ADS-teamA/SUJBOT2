class ChatWebSocketService {
  private ws: WebSocket | null = null;
  private messageHandlers: Array<(message: any) => void> = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private url: string = '';

  connect(url: string): void {
    this.url = url;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.messageHandlers.forEach(handler => handler(data));
      };

      this.ws.onerror = (error) => {
        // Suppress error logging during reconnect attempts
        if (this.reconnectAttempts === 0) {
          console.warn('WebSocket connection error (will retry)');
        }
      };

      this.ws.onclose = () => {
        if (this.reconnectAttempts === 0) {
          console.log('WebSocket disconnected');
        }
        this.handleReconnect();
      };
    } catch (error) {
      console.warn('Failed to create WebSocket, retrying...');
      this.handleReconnect();
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(() => this.connect(this.url), delay);
    }
  }

  sendMessage(message: string, documentIds?: string[]): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'chat_message',
        content: message,
        document_ids: documentIds || []
      }));
    }
  }

  onMessage(handler: (message: any) => void): void {
    this.messageHandlers.push(handler);
  }

  removeMessageHandler(handler: (message: any) => void): void {
    this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
  }

  disconnect(): void {
    this.ws?.close();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const chatWebSocket = new ChatWebSocketService();
