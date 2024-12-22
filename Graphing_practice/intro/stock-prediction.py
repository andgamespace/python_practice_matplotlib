import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        # Creating a deep neural network with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def prepare_data(symbol, lookback_days=30):
    """
    Prepares stock data with technical indicators and features
    """
    # Download maximum available data
    stock = yf.Ticker(symbol)
    df = stock.history(period="max")
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    # Create target variable (1 if price goes up next day, 0 if down)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Create features using the lookback period
    feature_columns = ['Returns', 'SMA_20', 'SMA_50', 'RSI', 'Volume_MA']
    X = []
    y = []
    
    for i in range(lookback_days, len(df)-1):
        X.append(df[feature_columns].iloc[i-lookback_days:i].values.flatten())
        y.append(df['Target'].iloc[i])
    
    return np.array(X), np.array(y)

def calculate_rsi(prices, periods=14):
    """
    Calculates Relative Strength Index
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(symbol, epochs=50, batch_size=32, learning_rate=0.001):
    # Prepare data
    X, y = prepare_data(symbol)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = StockPredictor(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    predicted = (outputs.data > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted.squeeze() == batch_y).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, '
                  f'Validation Accuracy: {accuracy:.2f}%')
    
    return model, scaler

def predict_next_day(model, scaler, symbol):
    """
    Predicts the probability of price increase for the next day
    """
    X, _ = prepare_data(symbol)
    # Get the most recent data point
    latest_data = X[-1:]
    # Scale the data
    scaled_data = scaler.transform(latest_data)
    # Convert to tensor
    data_tensor = torch.FloatTensor(scaled_data)
    
    model.eval()
    with torch.no_grad():
        prediction = model(data_tensor)
    
    return prediction.item()

# Example usage
if __name__ == "__main__":
    symbol = "AAPL"  # Example with Apple stock
    model, scaler = train_model(symbol)
    probability = predict_next_day(model, scaler, symbol)
    print(f"Probability of {symbol} price increasing tomorrow: {probability:.2f}")
