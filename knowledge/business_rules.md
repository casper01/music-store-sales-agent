# Fictional exercise rules: business definitions

> These are invented rules for this exercise. They are not official Chinook definitions or policies.

## Revenue

Revenue means the sum of `InvoiceLine.UnitPrice * InvoiceLine.Quantity` for all invoice lines in
scope. Use invoice-line values rather than counting invoices or summing a customer-level field.
Unless a question specifies otherwise, report the currency as stored in the source data and do not
combine currencies without conversion data. Chinook contains no refund ledger, so this definition
does not subtract refunds.

## Order

For this exercise, one order means one row in `Invoice`. An invoice can contain several invoice
lines. Do not count each line as a separate order.

## Active customer

For this exercise, an active customer is a customer with at least one invoice during the trailing
12 months ending on the latest invoice date present in the database. State that this is a
business-rule definition based on the data horizon, not a live-calendar status.
