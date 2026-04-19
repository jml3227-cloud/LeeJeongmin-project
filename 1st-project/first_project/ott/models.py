import oracledb
import pandas as pd


class PredictionModel:
    def __init__(self, user, password, dsn):
        self.user = user
        self.password = password
        self.dsn = dsn

    def connect(self):
        return oracledb.connect(
            user=self.user,
            password=self.password,
            dsn=self.dsn
        )

    def get_training_data(self):
        connection = self.connect()
        query = """
            SELECT u.USER_SEQ,
                   u.AGE_GROUP,
                   e.INCOME_GROUP,
                   e.FAMILY_TYPE,
                   s.MONTHLY_FEE_CODE,
                   s.HAS_AD_PLAN,
                   ub.USE_FREQUENCY,
                   ub.USED_LAST_WEEK,
                   ub.AVG_MIN_WEEKDAY,
                   ub.AVG_MIN_WEEKEND,
                   ub.SEARCH_VIEW,
                   ub.RECOMMEND_VIEW,
                   ub.BINGE_WATCH,
                    (SELECT COUNT(*)
                     FROM USER_OTT_SERVICE os
                     WHERE u.USER_SEQ = os.USER_SEQ) AS OTT_COUNT,
                    (SELECT COUNT(*)
                     FROM USER_DEVICE ud
                     WHERE u.USER_SEQ = ud.USER_SEQ) AS DEVICE_COUNT,
                    (SELECT COUNT(*)
                     FROM USER_CONTENT_TYPE uc
                     WHERE u.USER_SEQ = uc.USER_SEQ) AS CONTENT_DIVERSITY,
                    (SELECT COUNT(*)
                     FROM USER_CONTENT_TYPE uc
                     WHERE u.USER_SEQ = uc.USER_SEQ
                       AND uc.CONTENT_CODE = 3) AS WATCH_ORIGINAL,
                    (SELECT COUNT(*)
                     FROM USER_CONTENT_TYPE uc
                     WHERE u.USER_SEQ = uc.USER_SEQ
                       AND uc.CONTENT_CODE = 4) AS WATCH_MOVIE,
                    (SELECT COUNT(*)
                     FROM USER_CONTENT_TYPE uc
                     WHERE u.USER_SEQ = uc.USER_SEQ
                       AND uc.CONTENT_CODE = 5) AS WATCH_SHORTFORM
            FROM USERS U
            JOIN ECONOMY e ON u.USER_SEQ = e.USER_SEQ
            JOIN SUBSCRIPTION s ON u.USER_SEQ = s.USER_SEQ
            JOIN USAGE_BEHAVIOR ub ON u.USER_SEQ = ub.USER_SEQ
            WHERE s.MONTHLY_FEE_CODE != 8
              AND s.HAS_AD_PLAN IS NOT NULL
        """
        df = pd.read_sql(query, connection)
        connection.close()
        return df

    def insert_predictions(self, df):
        connection = self.connect()
        cursor = connection.cursor()
        try:
            for _, row in df.iterrows():
                cursor.execute(
                    """INSERT INTO PREDICTION_RESULT (USER_SEQ, FREQ_GROUP)
                       VALUES (:1, :2)""",
                    [int(row['USER_SEQ']), int(row['PRED_GROUP'])]
                )
            connection.commit()
            print(f'PREDICTION_RESULT 저장 완료: {len(df)}명')
        except Exception as e:
            connection.rollback()
            print(f'[ERROR] 롤백 처리됨: {e}')
            raise
        finally:
            cursor.close()
            connection.close()

    def get_group_counts(self):
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT FREQ_GROUP, COUNT(*)
            FROM PREDICTION_RESULT
            GROUP BY FREQ_GROUP
            ORDER BY FREQ_GROUP
        """)
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        result = {row[0]: row[1] for row in rows}
        return result

    def get_user_data(self, user_seq):
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT u.USER_SEQ,
                   e.FAMILY_TYPE,
                   s.MONTHLY_FEE_CODE,
                   ub.AVG_MIN_WEEKDAY,
                   ub.AVG_MIN_WEEKEND,
                   ub.SEARCH_VIEW,
                   ub.RECOMMEND_VIEW,
                   ub.BINGE_WATCH,
                   ub.USED_LAST_WEEK,
                   (SELECT COUNT(*)
                    FROM USER_OTT_SERVICE os
                    WHERE u.USER_SEQ = os.USER_SEQ) AS OTT_COUNT,
                   (SELECT COUNT(*)
                    FROM USER_DEVICE ud
                    WHERE u.USER_SEQ = ud.USER_SEQ) AS DEVICE_COUNT,
                   (SELECT COUNT(*)
                    FROM USER_CONTENT_TYPE uc
                    WHERE u.USER_SEQ = uc.USER_SEQ) AS CONTENT_DIVERSITY,
                   (SELECT COUNT(*)
                    FROM USER_CONTENT_TYPE uc
                    WHERE u.USER_SEQ = uc.USER_SEQ
                      AND uc.CONTENT_CODE = 3) AS WATCH_ORIGINAL,
                   (SELECT COUNT(*)
                    FROM USER_CONTENT_TYPE uc
                    WHERE u.USER_SEQ = uc.USER_SEQ
                      AND uc.CONTENT_CODE = 4) AS WATCH_MOVIE,
                   (SELECT COUNT(*)
                    FROM USER_CONTENT_TYPE uc
                    WHERE u.USER_SEQ = uc.USER_SEQ
                      AND uc.CONTENT_CODE = 5) AS WATCH_SHORTFORM
            FROM USERS u
            JOIN ECONOMY e ON u.USER_SEQ = e.USER_SEQ
            JOIN SUBSCRIPTION s ON u.USER_SEQ = s.USER_SEQ
            JOIN USAGE_BEHAVIOR ub ON u.USER_SEQ = ub.USER_SEQ
            WHERE u.USER_SEQ = :1
        """, [user_seq])

        row = cursor.fetchone()
        cursor.close()
        connection.close()
        if row is None:
            return None
        return {
            'USER_SEQ': row[0],
            'FAMILY_TYPE': row[1],
            'MONTHLY_FEE_CODE': row[2],
            'AVG_MIN_WEEKDAY': row[3],
            'AVG_MIN_WEEKEND': row[4],
            'SEARCH_VIEW': row[5],
            'RECOMMEND_VIEW': row[6],
            'BINGE_WATCH': row[7],
            'USED_LAST_WEEK': row[8],
            'OTT_COUNT': row[9],
            'DEVICE_COUNT': row[10],
            'CONTENT_DIVERSITY': row[11],
            'WATCH_ORIGINAL': row[12],
            'WATCH_MOVIE': row[13],
            'WATCH_SHORTFORM': row[14],
        }

    def get_max_user_seq(self):
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(USER_SEQ) FROM USERS")
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        return row[0]