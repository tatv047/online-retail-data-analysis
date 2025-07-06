# Data Dictionary

| Variable Name | Role | Type | Description |
|---------------|------|------|-------------|
| InvoiceNo | ID | Categorical | A 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation |
| StockCode | ID | Categorical | A 5-digit integral number uniquely assigned to each distinct product |
| Description | Feature | Categorical | Product name |
| Quantity | Feature | Integer | The quantities of each product (item) per transaction |
| InvoiceDate | Feature | Date | The day and time when each transaction was generated |
| UnitPrice | Feature | Continuous | Product price per unit |
| CustomerID | Feature | Categorical | A 5-digit integral number uniquely assigned to each customer |
| Country | Feature | Categorical | The name of the country where each customer resides |