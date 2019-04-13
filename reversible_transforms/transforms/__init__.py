import cat_transform as cat_n
import num_transform as num_n
import datetime_transform as datetime_n
import time_series_transform as time_series_n
import string_transform as string_n
import set_transform as set_n
import index_transform as index_n

CatTransform = cat_n.CatTransform
NumTransform = num_n.NumTransform
DateTimeTransform = datetime_n.DateTimeTransform
TimeSeriesTransform = time_series_n.TimeSeriesTransform
StringTransform = string_n.StringTransform
SetTransform = set_n.SetTransform
IndexTransform = index_n.IndexTransform

# Since using eval seems to screw up try - except blocks
str_to_obj =  {
  "CatTransform": cat_n.CatTransform,
  "NumTransform": num_n.NumTransform,
  "DateTimeTransform": datetime_n.DateTimeTransform,
  "TimeSeriesTransform": time_series_n.TimeSeriesTransform,
  "StringTransform": string_n.StringTransform,
  "SetTransform": set_n.SetTransform,
  "IndexTransform": index_n.IndexTransform
}
