from DataBase.database import engine, Base
from DataBase.models import Booking

# This will create the bookings table
Base.metadata.create_all(bind=engine)

print("✅ Bookings table created successfully!")
